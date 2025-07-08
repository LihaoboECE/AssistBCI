'''
    Cross session Test on SEED dataset - based on pura EEG (not DE feature)

    Feature extractor: EmoAdapt (self-supervised model)
    Classifier: SVM (default parameters)

    Author: Lihaobo
    Email: dc22799@umac.mo
    2025/7/3
'''

import sys
import os
import torch
import numpy as np
from metabci.brainda.datasets import SEED
from metabci.brainda.paradigms import Emotion
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from metabci.brainda.algorithms.self_supervised_learning.Base import TorchDataset

from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
import argparse
from metabci.brainda.algorithms.self_supervised_learning import EmoAdapt
from metabci.brainda.algorithms.self_supervised_learning.utils import plot_embedding
from sklearn.manifold import TSNE
from metabci.brainda.algorithms.self_supervised_learning.GMM import classifier
from metabci.brainda.algorithms.self_supervised_learning.GMM_distance import PCA_classifier
from sklearn import svm
from scipy import stats
import pandas as pd
from collections import defaultdict


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def relocate_data(X, Y, meta):
    """
    Relocate X and Y data according to meta information in order of:
    subject → session → run (from lower to higher values)

    Parameters:
    X (np.array): Experiment data array
    Y (np.array): Labels array
    meta (pd.DataFrame): Metadata DataFrame with subject, session, run info

    Returns:
    tuple: (X_sorted, Y_sorted, meta_sorted) - sorted versions of inputs
    """
    # Make copies to avoid modifying originals
    meta_sorted = meta.copy()

    # Convert subject to numeric if it's not already
    meta_sorted['subject'] = pd.to_numeric(meta_sorted['subject'])

    # Extract numeric parts from session and run for proper sorting
    meta_sorted['session_num'] = meta_sorted['session'].str.extract('(\d+)').astype(int)
    meta_sorted['run_num'] = meta_sorted['run'].str.extract('(\d+)').astype(int)

    # Sort the meta data
    meta_sorted = meta_sorted.sort_values(['subject', 'session_num', 'run_num', 'trial_id'])

    # Get the sorting indices
    sort_indices = meta_sorted.index

    # Apply the same sorting to X and Y
    X_sorted = X[sort_indices]
    Y_sorted = Y[sort_indices]

    # Drop the temporary numeric columns
    meta_sorted = meta_sorted.drop(columns=['session_num', 'run_num'])

    return X_sorted, Y_sorted, meta_sorted


def get_args(file_name):
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_fold', default=0, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', 'models_buf', file_name), type=str)
    return parser.parse_args()


dataset_path = 'E:\SEED'


train_dataset = SEED(path=dataset_path, win_duration=5, sessions=[0])

paradigm = Emotion(
    srate=200,
    channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
)

X_train, Y_train, _ = paradigm.get_data(
    train_dataset,
    # subjects=[i+1 for i in range(15)],
    subjects=[2],
    return_concat=True,
    n_jobs=5,
    verbose=False)


test_dataset = SEED(path=dataset_path, win_duration=5, sessions=[1])

paradigm = Emotion(
    srate=200,
    channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
)

X_test, Y_test, _ = paradigm.get_data(
    test_dataset,
    # subjects=[i+1 for i in range(15)],
    subjects=[2],
    return_concat=True,
    n_jobs=5,
    verbose=False)


args = get_args(file_name='EmoAdapt')


# 此模型为自监督模型, 无监督训练
model_path = os.path.join(args.ckpt_path, str(args.n_fold), 'model', 'best_model.pth')
model = torch.load(model_path, map_location='cpu', weights_only=False)['model'].to(device)

print("start feature extraction")
# 通过DataLoader节省显存, 也可以使用EmoAdapt.predict()直接预测
train_x, train_y = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
train_dataset = TorchDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)

test_x, test_y = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
test_dataset = TorchDataset(test_x, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

(latent_train, train_y), (latent_test, test_y) = model.get_latent(train_dataloader, disable_BN=True), model.get_latent(test_dataloader, disable_BN=True)

print("start T-sne")
tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
latent_tsne = tsne.fit_transform(latent_train)
plot_embedding(latent_tsne, train_y, "Session-2 t-SNE")

tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
latent_tsne = tsne.fit_transform(latent_test)
plot_embedding(latent_tsne, test_y, "Session-2 t-SNE")


print("start training")
classifier = svm.SVC()
classifier.fit(latent_train, train_y)
out = classifier.predict(latent_test)
acc = np.mean(out == test_y)
print("ACC: ", acc)

