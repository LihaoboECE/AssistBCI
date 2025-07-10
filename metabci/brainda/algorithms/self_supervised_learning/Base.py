import os.path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset
from typing import List
from timm.models.vision_transformer import Block
from functools import partial
import torch.nn.functional as F
from skorch.classifier import NeuralNetClassifier
from collections import OrderedDict
import torch.optim as optim
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, EpochScoring, Checkpoint, EarlyStopping



'''
1D simplified Resnet Backbone
'''

class SimplifiedResnet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.model = SimplifiedBackbone(input_size=size, input_channel=1, layers=[1, 1, 1])  # 减少层数
        self.feature_layer = nn.Sequential(
            nn.Linear(self.model.get_final_length(), size),
            nn.ELU(),
        )

    def forward(self, x):
        latent_seq = []
        for i in range(x.shape[1]):
            sample = torch.unsqueeze(x[:, i, :], dim=1)
            latent = self.model(sample)
            latent_seq.append(latent)
        latent_seq = torch.stack(latent_seq, dim=1)
        latent_seq = self.feature_layer(latent_seq)
        return latent_seq

class SimplifiedBackbone(nn.Module):
    def __init__(self, input_size, input_channel, layers):
        super().__init__()
        self.inplanes3 = 16  # 3x3分支的初始通道数
        self.inplanes5 = 16  # 5x5分支的初始通道数
        self.input_size = input_size

        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channel, 16, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 3x3分支
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 16, layers[0], stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 24, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 32, layers[2], stride=2)
        self.maxpool3 = nn.AdaptiveAvgPool1d(32)

        # 5x5分支
        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 16, layers[0], stride=1)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 24, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 32, layers[2], stride=2)
        self.maxpool5 = nn.AdaptiveAvgPool1d(32)


    def forward(self, x0):
        b = x0.shape[0]
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        # x0 = torch.transpose(x0, 2, 0)  # 转置以适应Conv1d输入格式

        # 3x3分支处理
        x1 = self.layer3x3_1(x0)
        x1 = self.layer3x3_2(x1)
        x1 = self.layer3x3_3(x1)
        x1 = self.maxpool3(x1)

        # 5x5分支处理
        x2 = self.layer5x5_1(x0)
        x2 = self.layer5x5_2(x2)
        x2 = self.layer5x5_3(x2)
        x2 = self.maxpool5(x2)

        out = torch.cat([x1, x2], dim=1)
        out = torch.reshape(out, [b, -1])
        return out

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def get_final_length(self):
        x = torch.randn(1, 1, int(self.input_size))
        x = self.forward(x)
        return x.shape[-1]


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 确保残差连接形状匹配
        if out.shape[2] != residual.shape[2]:
            # 计算需要填充或裁剪的量
            diff = residual.shape[2] - out.shape[2]
            residual = residual[:, :, :out.shape[2]] if diff > 0 else F.pad(residual, (0, -diff))

        out += residual
        out = self.relu(out)
        return out


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 确保残差连接形状匹配
        if out.shape[2] != residual.shape[2]:
            # 计算需要填充或裁剪的量
            diff = residual.shape[2] - out.shape[2]
            residual = residual[:, :, :out.shape[2]] if diff > 0 else F.pad(residual, (0, -diff))

        out += residual
        out = self.relu(out)
        return out



'''
MAE Model
'''
class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, input_size: int, num_patches: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 ):
        super().__init__()
        self.patch_embed = nn.Linear(input_size, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.embed_dim = encoder_embed_dim
        self.encoder_depths = encoder_depths
        self.mlp_ratio = 4.

        self.input_size = (num_patches, encoder_embed_dim)
        self.patch_size = (1, encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w

        # MAE Encoder
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, encoder_embed_dim), requires_grad=False)
        self.encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # MAE Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, int(self.num_patches), decoder_embed_dim), requires_grad=False)
        self.decoder_block = nn.ModuleList([
            Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depths)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, input_size, bias=True)
        self.initialize_weights()

    def forward(self, x, mask_ratio=0.8):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return latent, pred, mask

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.5):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x[:, 1:, :])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        index = ids_restore.unsqueeze(-1)
        index = index.repeat(1, 1, x.shape[2])
        x = torch.gather(x_, dim=1, index=index)  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.decoder_block:
            x = block(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape  # batch, length, dim
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_shuffle = ids_shuffle % l
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index = ids_keep.unsqueeze(-1).repeat(1, 1, d)
        x_masked = torch.gather(x, dim=1, index=index)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                     (self.grid_h, self.grid_w),
                                                     cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1],
                                                             (self.grid_h, self.grid_w),
                                                             cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_sizes[0], dtype=np.float32)
    grid_w = np.arange(grid_sizes[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_sizes[0], grid_sizes[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


'''
NTXent Loss
'''
class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    @staticmethod
    def mask_correlated_samples(batch_size):
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=bool)
        mask = mask.fill_diagonal_(0)  #self distance -> 0

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        # t = torch.cat((t, t), dim=0)
        # time_sim = torch.abs(t.unsqueeze(0) - t.unsqueeze(1))
        # time_sim = torch.log10(time_sim)*0.1 + 1

        batch_size = z_j.shape[0]
        n = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = f.normalize(z, dim=-1)

        mask = self.mask_correlated_samples(batch_size)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        negative_samples = sim[mask].reshape(n, -1)
        # negative_samples = (sim[mask]*time_sim[mask]).reshape(n, -1)


        labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(positive_samples.device).long()  # .float()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature

        loss = self.criterion(logits, labels)
        loss /= n
        return loss, (labels, logits)

# class NTXentLoss(nn.Module):
#     def __init__(self, temperature, num_per_group=4):
#         super().__init__()
#         self.criterion = nn.CrossEntropyLoss(reduction='sum')
#         self.similarity_f = nn.CosineSimilarity(dim=-1)
#         self.temperature = temperature
#         self.num_per_group = num_per_group
#         self.feature_layer = nn.Sequential(
#             nn.ELU(),
#             nn.Linear(512, 1),
#         )
#
#     # @staticmethod
#     def mask_correlated_samples(self, batch_size):
#         n = 2 * batch_size
#         mask = torch.ones((n, n), dtype=bool)
#         mask = mask.fill_diagonal_(0)  #self distance -> 0
#
#         for i in range(batch_size):
#
#             # if i % self.num_per_group == 0:
#             #     mask[i, batch_size + i] = 0
#             #     mask[batch_size + i, i] = 0
#
#             mask[i, batch_size + i] = 0
#             mask[batch_size + i, i] = 0
#         return mask
# #
#     def forward(self, z_i, z_j):
        # batch_size = z_j.shape[0]
        # n = batch_size
        # z_i, z_j = f.normalize(z_i, dim=-1), f.normalize(z_j, dim=-1)

        # loss = (z_i - z_j) ** 2
        # loss = loss.mean(dim=-1)
        # loss = torch.pow(0.2, loss-1)
        # loss = loss.sum()
        #
        # labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(z_i.device).long()  # .float()
        # logits = labels
        #
        # return torch.round(loss * 1e-2 * 10000) / 10000, (labels, logits)
#
        ##############################################################################
        # batch_size = z_j.shape[0]
        # n = batch_size
        #
        # z_i, z_j = self.feature_layer(z_i), self.feature_layer(z_j)
        #
        # labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(z_i.device).long()  # .float()
        #
        # logits = torch.cat((z_i,z_j), dim=1) / self.temperature
        # loss = self.criterion(logits, labels)
        # loss /= n
        # return loss*10, (labels, logits)
        # # return loss*100, (labels, logits)

        ##############################################################################

        # batch_size = z_j.shape[0]
        # n = int(2 * batch_size / self.num_per_group)
        # z = torch.cat((z_i, z_j), dim=0)
        # z = f.normalize(z, dim=-1)
        #
        # mask = self.mask_correlated_samples(batch_size)
        # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        #
        # sim_i_j = torch.diag(sim, batch_size)
        # sim_j_i = torch.diag(sim, -batch_size)
        #
        # sim_i_j = sim_i_j[::self.num_per_group]
        # sim_j_i = sim_j_i[::self.num_per_group]
        #
        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        # negative_samples = sim[mask].reshape(n, -1)
        # # negative_samples = (sim[mask]*time_sim[mask]).reshape(n, -1)
        #
        #
        # labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(positive_samples.device).long()  # .float()
        # logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        #
        # loss = self.criterion(logits, labels)
        # loss /= n
        # return loss * 1e-1, (labels, logits)

        ##############################################################################

        # batch_size = z_j.shape[0]
        #
        # z_i, z_j = self.feature_layer(z_i), self.feature_layer(z_j)
        #
        # p_z_i, p_z_j = z_i[::self.num_per_group], z_j[::self.num_per_group]
        #
        # mask = torch.arange(batch_size) % self.num_per_group != 0
        # n_z_i, n_z_j = z_i[mask], z_j[mask]
        #
        # p_loss = (p_z_i - p_z_j) ** 2
        # p_loss = p_loss.sum()
        #
        # labels = torch.from_numpy(np.array([0] * n_z_i.shape[0])).reshape(-1).to(z_i.device).long()  # .float()
        # logits = torch.cat((n_z_i, n_z_j), dim=1)
        # n_loss = self.criterion(logits, labels)
        # loss = (p_loss/ p_z_i.shape[0]) + (n_loss / n_z_i.shape[0])
        #
        # return loss, (labels, logits)

        ##############################################################################

        # batch_size = z_j.shape[0]
        # n = batch_size
        #
        # z_i, z_j = torch.mean(z_i, dim=1), torch.mean(z_j, dim=1)
        # z_i, z_j = z_i.unsqueeze(-1), z_j.unsqueeze(-1)
        #
        # labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(z_i.device).long()  # .float()
        #
        # logits = torch.cat((z_i, z_j), dim=1)
        # loss = self.criterion(logits, labels)
        # loss /= n
        # return loss * 10, (labels, logits)
        # return loss*100, (labels, logits)


class TorchDataset(Dataset):
    def __init__(self, *arrays):
        """
        Initialize the dataset with multiple arrays.

        Args:
            *arrays: Variable number of input arrays (x, y, x_o, y_o, etc.)
                      All arrays must have the same length.
        """
        # Check that all arrays have the same length
        if len(arrays) > 0:
            first_len = len(arrays[0])
            for i, arr in enumerate(arrays):
                if len(arr) != first_len:
                    raise ValueError(f"All arrays must have the same length. "
                                     f"Array 0 has length {first_len}, but array {i} has length {len(arr)}")

        self.arrays = arrays

    def __len__(self):
        """Return the length of the dataset (length of the first array)."""
        if len(self.arrays) == 0:
            return 0
        return len(self.arrays[0])

    def __getitem__(self, index):
        """
        Get the items at the given index from all arrays.

        Args:
            index: Index of the items to retrieve

        Returns:
            Tuple containing the items at the given index from each array
        """
        if len(self.arrays) == 0:
            raise IndexError("No arrays in the dataset")

        # Get the item from each array
        items = []
        for arr in self.arrays:
            # Convert to tensor if it's not already one
            if not isinstance(arr, torch.Tensor):
                item = torch.tensor(arr[index])
            else:
                item = arr[index]
            items.append(item)

        # Return as tuple if multiple arrays, otherwise return single item
        if len(items) == 1:
            return items[0]
        return tuple(items)