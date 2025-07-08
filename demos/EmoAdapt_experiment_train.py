import sys
sys.path.append('/root/metabci')
import os
import torch
import numpy as np
from metabci.brainda.datasets import SEED, unLabeled_EEG
from metabci.brainda.paradigms import Emotion
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
import argparse
from metabci.brainda.algorithms.self_supervised_learning import EmoAdapt
from metabci.brainda.algorithms.self_supervised_learning.utils import plot_embedding
from sklearn.manifold import TSNE
from scipy import signal


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--k_splits', default=5)
#     parser.add_argument('--n_fold', default=0, type=int)
#     parser.add_argument('--fs', default=200, type=int)
#
#     # Train Hyperparameter
#     parser.add_argument('--seed', default=777, type=int)  # 30
#     parser.add_argument('--train_epochs', default=300, type=int) #30
#     parser.add_argument('--train_learning_rate', default=5e-5, type=float) #5e-5
#     parser.add_argument('--train_batch_size', default=32, type=int)
#
#     # Model Hyperparameter
#     parser.add_argument('--channels', default=8, type=int)
#     parser.add_argument('--second', default=5, type=int)
#     parser.add_argument('--time_window', default=1, type=int)
#     parser.add_argument('--time_step', default=0.2, type=int)
#
#     parser.add_argument('--encoder_embed_dim', default=256, type=int)
#     parser.add_argument('--encoder_heads', default=4, type=int)
#     parser.add_argument('--encoder_depths', default=2, type=int)
#     parser.add_argument('--decoder_embed_dim', default=256, type=int)
#     parser.add_argument('--decoder_heads', default=4, type=int)
#     parser.add_argument('--decoder_depths', default=2, type=int) #3
#     parser.add_argument('--alpha', default=1, type=float)
#
#     parser.add_argument('--projection_hidden', default=[512, 256], type=list)#[1024, 512]
#     parser.add_argument('--temperature', default=0.05, type=float) #0.05
#     parser.add_argument('--mask_ratio', default=0.8, type=float)
#     parser.add_argument('--print_point', default=5, type=int)
#     parser.add_argument('--ckpt_path', default=os.path.join('..', 'models'), type=str)
#     parser.add_argument('--model_name', default='EmoAdapt')
#     return parser.parse_args()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, type=int)
    parser.add_argument('--fs', default=200, type=int)

    # Train Hyperparameter
    parser.add_argument('--seed', default=777, type=int)  # 30
    parser.add_argument('--train_epochs', default=300, type=int) #30
    parser.add_argument('--train_learning_rate', default=125e-6, type=float) #5e-5
    parser.add_argument('--train_batch_size', default=64, type=int)

    # Model Hyperparameter
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--second', default=5, type=int)
    parser.add_argument('--time_window', default=1, type=int)
    parser.add_argument('--time_step', default=0.2, type=int)

    parser.add_argument('--encoder_embed_dim', default=768, type=int)
    parser.add_argument('--encoder_heads', default=8, type=int)
    parser.add_argument('--encoder_depths', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=256, type=int)
    parser.add_argument('--decoder_heads', default=8, type=int)
    parser.add_argument('--decoder_depths', default=3, type=int) #3
    parser.add_argument('--alpha', default=1, type=float)

    parser.add_argument('--projection_hidden', default=[1024, 512], type=list)#[1024, 512]
    parser.add_argument('--temperature', default=0.05, type=float) #0.05
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--print_point', default=5, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', 'models'), type=str)
    parser.add_argument('--model_name', default='EmoAdapt')
    return parser.parse_args()

# dataset_path = "/home/ubuntu-user/Documents/dc22799_DoNotDeleteTanks/SEED"
# dataset = SEED(path=dataset_path, win_duration=5, sessions=[0,1])
#
# events = dataset.events.keys()
#
# paradigm = Emotion(
#     srate=200,
#     channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
# )
#
# X, y, meta = paradigm.get_data(
#     dataset,
#     # subjects=[i+1 for i in range(15)],
#     subjects=[1],
#     return_concat=True,
#     n_jobs=5,
#     verbose=False)
#
# args = get_args()
#
# kfold = 5
# indices = generate_kfold_indices(meta, kfold=kfold)
#
# for k in range(kfold):
#     train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
#     # run = meta["run"].values
#     # session = meta["session"].values
#     # r1, r2, r3 = run[train_ind], run[validate_ind], run[test_ind]
#     # s1, s2, s3 = session[train_ind], session[validate_ind], session[test_ind]
#     args.n_fold = k
#
#     model = EmoAdapt(fs=args.fs, second=args.second, time_window=args.time_window, time_step=args.time_step,
#                      encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
#                      encoder_depths=args.encoder_depths,
#                      decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
#                      decoder_depths=args.decoder_depths,
#                      channels=args.channels, train_batch_size=args.train_batch_size,
#                      lr=args.train_learning_rate, train_epochs=args.train_epochs,
#                      mask_ratio=args.mask_ratio, alpha=args.alpha,
#                      print_point=args.print_point, device=device, random_seed=args.seed,
#                      projection_hidden=args.projection_hidden, temperature=args.temperature,
#                      model_path=args.ckpt_path, model_name=args.model_name, n_fold=args.n_fold)
#
#     model.fit(X[train_ind], y[train_ind], X[validate_ind], y[validate_ind])
#
#     latent = model.predict(X[train_ind])
#
#     tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
#     latent_tsne = tsne.fit_transform(latent)
#     plot_embedding(latent_tsne, y, "Session-2 t-SNE")



##cross session

train_dataset = unLabeled_EEG(data_paths=['E:/2025comp/7_7datasetLHB/PPE-1.edf', 'E:/2025comp/7_7datasetLHB/PPE-2.edf'], win_duration=5, drate=200, D=5)

paradigm = Emotion(
    srate=200,
    channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
)

x, y, _ = paradigm.get_data(
    train_dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=1,
    verbose=False)

args = get_args()

x = signal.detrend(x)
# 50Hz滤波器
b, a = signal.iirnotch(50, 4, 200)
x = signal.filtfilt(b, a, x)

# 1~75 butterworth 减少相位失真
low = 1 / 100
high = 75 / 100
b, a = signal.butter(4, [low, high], btype='band')
x = signal.filtfilt(b, a, x)

x = x.copy()


model = EmoAdapt(fs=args.fs, second=args.second, time_window=args.time_window, time_step=args.time_step,
                 encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
                 encoder_depths=args.encoder_depths,
                 decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
                 decoder_depths=args.decoder_depths,
                 channels=args.channels, train_batch_size=args.train_batch_size,
                 lr=args.train_learning_rate, train_epochs=args.train_epochs,
                 mask_ratio=args.mask_ratio, alpha=args.alpha,
                 print_point=args.print_point, device=device, random_seed=args.seed,
                 projection_hidden=args.projection_hidden, temperature=args.temperature,
                 model_path=args.ckpt_path, model_name=args.model_name, n_fold=args.n_fold)

model.fit(x, y, x, y)

