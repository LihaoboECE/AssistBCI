'''
    Online study Test on SEED dataset

    Feature extractor: EmoAdapt (self-supervised model)
    Classifier: Smooth-Prototype (Proposed online study classifier for emotion/physiological status)

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
import argparse
from metabci.brainda.algorithms.self_supervised_learning.utils import plot_embedding
from sklearn.manifold import TSNE
from metabci.brainda.algorithms.self_supervised_learning.prototype import PCA_classifier
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
    parser.add_argument('--ckpt_path', default=os.path.join('..', 'assistbci_models', file_name), type=str)
    return parser.parse_args()


dataset_path = 'E:\SEED'
dataset = SEED(path=dataset_path, win_duration=5, sessions=[2])

paradigm = Emotion(
    srate=200,
    # channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]  # for 8-ch model
    channels=["FP1", "C5", "CP3", "P4"]                            # for 4-ch model
)

X, Y, meta = paradigm.get_data(
    dataset,
    subjects=[15],
    return_concat=True,
    n_jobs=5,
    verbose=False)

X, Y, meta = relocate_data(X, Y, meta)

args = get_args(file_name='EmoAdapt')

model_path = os.path.join(args.ckpt_path, str(args.n_fold), 'model', 'best_model.pth')
model = torch.load(model_path, map_location='cpu', weights_only=False)['model'].to(device)


# only for brief view of data
_X = model.predict(X)
tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
latent_tsne = tsne.fit_transform(_X)
plot_embedding(latent_tsne, Y, "Session-2 t-SNE")


# 创建初始GMM模型（1类）
print("\n##########################模型初始化##########################")
gmm = PCA_classifier(n_features=10, pca_patch_max_len=20, max_buf_size=50,
                 min_samples_per_class=20, adjustment_step=0.5, max_adjustment=10.0)
X_1 = X[Y==1]
np.random.shuffle(X_1)
X_1 = X_1[0:12]
X_1 = model.predict(X_1)

gmm.fit(X_1, np.ones(X_1.shape[0], dtype=int))  # 初始类为1 (中性neutral)

# 实验参数
MIN_DETECTION_INTERVAL = 12  # 最小检测间隔（样本数）
MIN_ACC_NEW_CLASS = 3
MAX_BUFFER_SIZE = 3  # 最大缓存样本数
stop_update_after = 2000 / 5
TEST = False # Enable this for offline test after online time
experiment_time_for_each_trail = int(120 / 5)

# 开始实验
global_cont = 0
print("\n##########################实验开始##########################")
while X.shape[0] >= experiment_time_for_each_trail:
    new_data, new_label, buffer = [], [], []  # new_data/label：新类数据
    interrupt_interval = 0
    for i in range(experiment_time_for_each_trail):
        x = X[i, ...]

        x = model.predict(x[np.newaxis, ...])[0, ...]

        _ground_true = Y[i]
        interrupt_interval += 1
        print("Ground Truth: ", _ground_true)
        gmm.sample_counter += 1
        global_cont += 1

        if global_cont == stop_update_after:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!停止模型操作，仅接受预测!!!!!!!!!!!!!!!!!!!!!!!!!!")

        new, labels, report = gmm.predict(x)
        new, labels, report = new[0], labels[0], report[0]

        if new and global_cont < stop_update_after:
            # 维护固定大小的缓存
            buffer.append(x)
            if len(buffer) > MAX_BUFFER_SIZE:
                buffer.pop(0)

            # 多级检测决策
            if gmm.should_trigger_new_class_detection(labels[0], buffer, MIN_DETECTION_INTERVAL, MIN_ACC_NEW_CLASS):
                # 处理新类
                buffer_array = np.stack(buffer)
                new_data.append(buffer_array)
                new_label.append(np.array([Y[i] for _ in range(buffer_array.shape[0])]))
                print(f"#############################检测到新状态! Auto class: {Y[i]}#############################")
                print("距离上次通知时长：", interrupt_interval * 5, "sec / ", interrupt_interval * 5 / 60, "min\n")
                interrupt_interval = 0

                # 重置状态
                gmm.reset_detection_state(labels[0])
                buffer = []
                break
        else:
            # 重置缓存
            buffer = []
            gmm.reset_detection_state(labels[0])

            # 处理错误反馈
            if report:
                print("状态预测  》》》》 预测状态：", labels, "真实状态：", Y[i])
                # if len(labels) == 1:
                #     new_data.append(np.array(x)[np.newaxis, :])
                #     new_label.append(np.array([labels[0]]))

                if global_cont < stop_update_after and labels[0] != Y[i]:
                    gmm.error_pred_feedback(labels[0])
                    print("报告错误识别")

    #清除用过的数据，整理数据
    X, Y = X[i:, :], Y[i:]

    if not new_data:
        print("未检测到新类，线上实验继续")
        continue

    new_data = np.concatenate(new_data, axis=0)
    new_label = np.concatenate(new_label, axis=0)

    #重新训练模型，确定阈值
    gmm.fit(new_data, new_label)

    if TEST:
        # 测试准确率
        print("\n-----测试开始-----")
        new_class, pred_classes, report = gmm.predict(model.predict(X[0:experiment_time_for_each_trail, :]))
        print("预测存在新类：", np.sum(new_class), "次") #最好为0次
        print("报告状态：", np.sum(report), "次")

        non_zero_report = 0
        length_counts = defaultdict(int)
        for i, sublist in enumerate(pred_classes):
            if not report[i]:
                continue
            length = len(sublist)
            length_counts[length] += 1
            if sublist[0] != 0:
                non_zero_report += 1
        print("预测状态同时存在n个的有几次", dict(length_counts))  # 输出: {1: 1, 2: 1, 3: 1}
        print("Non zero report times: ", non_zero_report)

        acc = np.mean(np.array([sublist[0] for sublist in pred_classes]) == Y[0:experiment_time_for_each_trail])
        print("ACC: ", acc)

        if np.sum(report):
            report_pred = np.array([sublist[0] for i, sublist in enumerate(pred_classes) if report[i]])
            report_pred_ground_truth = np.array([y for i, y in enumerate(Y[0:experiment_time_for_each_trail]) if report[i]])
            report_acc = np.mean(report_pred == report_pred_ground_truth)
            print("Report ACC: ", report_acc)
        else:
            print("skip report acc")

        X, Y = X[experiment_time_for_each_trail:, :], Y[experiment_time_for_each_trail:]

        print("-----测试结束-----\n")

        if acc >= 0.99:
            print("保存特征长度：", len(gmm.buf_labels))

            print("\n-----最终测试开始-----")
            new_class, pred_classes, report = gmm.predict(X)
            print("预测存在新类：", np.sum(new_class), "次")  # 最好为0次
            print("报告状态：", np.sum(report), "次")

            non_zero_report = 0
            length_counts = defaultdict(int)
            for i, sublist in enumerate(pred_classes):
                if not report[i]:
                    continue
                length = len(sublist)
                length_counts[length] += 1
                if sublist[0] != 0:
                    non_zero_report += 1
            print("预测状态同时存在n个的有几次", dict(length_counts))  # 输出: {1: 1, 2: 1, 3: 1}
            print("Non zero report times: ", non_zero_report)

            acc = np.mean(np.array([sublist[0] for sublist in pred_classes]) == Y)
            print("ACC: ", acc)

            if np.sum(report):
                report_pred = np.array([sublist[0] for i, sublist in enumerate(pred_classes) if report[i]])
                report_pred_ground_truth = np.array(
                    [y for i, y in enumerate(Y) if report[i]])
                report_acc = np.mean(report_pred == report_pred_ground_truth)
                print("Report ACC: ", report_acc)
            else:
                print("skip report acc")
            print("-----最终测试结束-----\n")
            break

print("\n##########################实验结束##########################")

