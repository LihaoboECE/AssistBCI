# -*- coding:utf-8 -*-
import os.path
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import copy
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
from metabci.brainda.algorithms.self_supervised_learning.Base import MaskedAutoEncoderViT, NTXentLoss, TorchDataset, SimplifiedResnet
from metabci.brainda.algorithms.self_supervised_learning.utils import augment_data



'''
EmoAdapt Structure
'''
class EmoAdapt(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 channels: int, projection_hidden: List, temperature: float,
                 lr: float, train_batch_size: int, train_epochs: int, mask_ratio: float, alpha: float,
                 print_point: int, random_seed: int, n_fold: int, device: torch.device,
                 model_path: str, model_name: str):

        super().__init__()
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = device
        self.n_fold = n_fold
        self.model_name = model_name
        self.model_path = model_path

        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step
        self.channels = channels

        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.print_point = print_point
        self.temperature = temperature


        self.mask_ratio = mask_ratio
        self.alpha = alpha

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_heads = encoder_heads
        self.encoder_depths = encoder_depths
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_heads = decoder_heads
        self.decoder_depths = decoder_depths
        self.projection_hidden = projection_hidden

        self.num_patches, _ = token_len(fs=fs, second=second, time_window=time_window, time_step=time_step, channels=channels)

        self.backbone = SimplifiedResnet(size=self.fs * self.time_window)

        self.autoencoder = MaskedAutoEncoderViT(input_size=self.fs*self.time_window,
                                                encoder_embed_dim=encoder_embed_dim, num_patches=self.num_patches,
                                                encoder_heads=encoder_heads, encoder_depths=encoder_depths,
                                                decoder_embed_dim=decoder_embed_dim, decoder_heads=decoder_heads,
                                                decoder_depths=decoder_depths)

        self.contrastive_loss = NTXentLoss(temperature=temperature)

        projection_hidden = [encoder_embed_dim] + projection_hidden
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            if i != len(projection_hidden) - 2:
                projectors.append(nn.Linear(h1, h2))
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
                projectors.append(nn.Dropout(p=0.5))
            else:
                projectors.append(nn.Linear(h1, h2))
        self.projectors = nn.Sequential(*projectors)
        self.projectors_bn = nn.BatchNorm1d(projection_hidden[-1], affine=False)


        self.lr = lr

        self.to(self.device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_epochs)


    def forward(self, data_aug: torch.Tensor, data: torch.Tensor, mask_ratio: float = 0.5) -> (torch.Tensor, torch.Tensor):

        x = data[:, 0:self.channels, :] # for 2 channels input only

        x = self.make_token(x)
        data_frame = self.backbone(x)

        data_aug = self.make_token(data_aug)
        data_aug = self.backbone(data_aug)

        # Masked Prediction
        latent_ref, _ , _ = self.autoencoder.forward_encoder(data_aug, mask_ratio=0.5)
        latent, pred, mask = self.autoencoder(data_frame, mask_ratio)

        o, o_ref = latent[:, :1, :].squeeze(), latent_ref[:, :1, :].squeeze()
        o1, o2 = self.projectors(o), self.projectors(o_ref)
        contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)

        # Contrastive Learning
        recon_loss1 = self.forward_mae_loss(x, pred, mask)

        return recon_loss1, contrastive_loss, (labels, logits)


    def forward_predict(self, x: torch.Tensor):
        x = self.make_token(x)
        x = self.backbone(x)
        latent, _, _ = self.autoencoder.forward_encoder(x, mask_ratio=0)
        latent_o = latent[:, :1, :].squeeze()
        if len(latent_o.shape) == 1:
            latent_o = latent_o.unsqueeze(0)
        return latent_o


    def predict(self, x: np.ndarray, disable_BN=True):
        # predict in one turn
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if disable_BN:
            self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            latent = self.forward_predict(x)
        if disable_BN:
            self.train()
        return latent.detach().cpu().numpy()


    def get_latent(self, dataloader, disable_BN=False):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                try:
                    x, y = data
                except:
                    x, _, y, _ = data
                x, y = x.to(self.device), y.to(self.device)
                # start_time = time.time()
                if disable_BN:
                    self.eval()
                latent = self.forward_predict(x)
                if disable_BN:
                    self.train()
                # print("predicting time for each batch: ", time.time()-start_time)
                if len(latent.shape) == 1:
                    latent = latent.unsqueeze(0)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)
        return total_x, total_y


    def ML_probing(self, train_dataloader, val_dataloader):
        self.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent(train_dataloader), \
            self.get_latent(val_dataloader)

        # tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
        # latent_tsne = tsne.fit_transform(test_x)
        # plot_embedding(latent_tsne, test_y, "Session-2 t-SNE")

        model = MLPClassifier()
        model.fit(train_x, train_y)
        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.train()
        return acc, mf1


    def fit(self, x: np.ndarray, y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray):
        aug_x, origin_x, original_y = augment_data(x, y)

        aug_x, origin_x, original_y = torch.tensor(aug_x, dtype=torch.float32), \
            torch.tensor(origin_x, dtype=torch.float32), torch.tensor(original_y, dtype=torch.long)

        train_dataset = TorchDataset(aug_x, origin_x, original_y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

        train_x, train_y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        train_val_dataset = TorchDataset(train_x, train_y)
        train_val_dataloader = DataLoader(train_val_dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=False)

        validation_x, validation_y = torch.tensor(validation_x, dtype=torch.float32), torch.tensor(validation_y, dtype=torch.long)
        val_dataset = TorchDataset(validation_x, validation_y)
        val_dataloader = DataLoader(val_dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=False)

        total_step = 0
        best_model, best_score = copy.deepcopy(self), 0
        self.save_model(model=best_model)

        val_acc, val_mf1 = self.ML_probing(train_val_dataloader, val_dataloader)

        print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
            -1, val_acc * 100, val_mf1 * 100))

        for epoch in range(self.train_epochs):
            step = 0
            self.train()

            self.optimizer.zero_grad()

            for x, x_o, _ in train_dataloader:
                x, x_o = x.to(self.device), x_o.to(self.device)

                out = self.forward(x, x_o, mask_ratio=self.mask_ratio)
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = out

                loss = recon_loss + self.alpha * contrastive_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (total_step + 1) % self.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                          '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                          '[Total Loss] : {4:02.4f}  [Contrastive Acc] : {5:02.4f}'.format(
                        epoch, total_step + 1, recon_loss, contrastive_loss, loss,
                        self.compute_metrics(cl_logits, cl_labels)))

                step += 1
                total_step += 1

            if (epoch + 1) % 1 == 0:
                val_acc, val_mf1 = self.ML_probing(train_val_dataloader, val_dataloader)

                if val_mf1 > best_score:
                    best_model = copy.deepcopy(self)
                    best_score = val_mf1

                print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                    epoch, val_acc * 100, val_mf1 * 100))

                self.optimizer.zero_grad()
                self.scheduler.step()

                self.save_model(model=best_model)


    def make_token(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1).view(x.shape[0], -1, window)
        return frame


    @staticmethod
    def forward_mae_loss(real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        mean = real.mean(dim=-1, keepdim=True)
        var = real.var(dim=-1, keepdim=True)
        real = (real - mean) / (var + 1.e-6) ** .5

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy


    def save_model(self, model):
        model_path = os.path.join(self.model_path, self.model_name, str(self.n_fold), 'model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save({
            'model_name': self.model_name,
            'model': model,
            'model_parameter': {
                'fs': self.fs, 'second': self.second,
                'time_window': self.time_window, 'time_step': self.time_step,
                'encoder_embed_dim': self.encoder_embed_dim, 'encoder_heads': self.encoder_heads,
                'encoder_depths': self.encoder_depths,
                'decoder_embed_dim': self.decoder_embed_dim, 'decoder_heads': self.decoder_heads,
                'decoder_depths': self.decoder_depths,
                'projection_hidden': self.projection_hidden, 'temperature': self.temperature
            }
        }, os.path.join(model_path, 'best_model.pth'))


def token_len(fs, second, time_window, time_step, channels=2):
    x = np.random.randn(channels, fs * second)
    size = fs * second
    step = int(time_step * fs)
    window = int(time_window * fs)
    frame = []
    for i in range(0, size, step):
        start_idx, end_idx = i, i + window
        sample = x[..., start_idx: end_idx]
        if sample.shape[-1] == window:
            frame.append(sample)
    frame = np.stack(frame, axis=1)
    return frame.shape[0] * frame.shape[1], frame.shape[2]




