import os
import zipfile
from typing import Union, Optional, Dict, List, cast
from pathlib import Path
from collections import Counter

import numpy as np
from mne import create_info
from mne.io import RawArray, Raw
from mne.channels import make_standard_montage
from .base import BaseDataset
from metabci.brainda.utils.download import mne_data_path
import scipy.io
import pandas as pd

'''
code for reading SEED dataset
this code needs local SEED dataset
dataset can be find: https://bcmi.sjtu.edu.cn/home/seed/

**Important note:
this code can only load the 'Preprocessed_EEG' from SEED,
features Eg. DE, PSD is not available 

Author: Li Haobo
Email: lihaoboece@gmail.com
2025/7/8
'''


def time_str_to_seconds(time_str):
    """将时间字符串(HH:MM:SS或MM:SS)转换为总秒数"""
    if pd.isna(time_str):  # 处理空值
        return 0

    # 统一转换为字符串处理
    time_str = str(time_str)

    # 分割时间部分
    parts = time_str.split(':')

    try:
        if len(parts) == 3:  # HH:MM:SS格式
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
        elif len(parts) == 2:  # MM:SS格式
            hours = 0
            minutes = int(parts[0])
            seconds = int(parts[1])
        else:
            return 0  # 未知格式

        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0  # 转换失败


def calculate_duration(start_time, end_time):
    """计算两个时间点之间的秒数差"""
    start_sec = time_str_to_seconds(start_time)
    end_sec = time_str_to_seconds(end_time)
    return end_sec - start_sec  # 确保结果不为负



class SEED(BaseDataset):

    def __init__(self, path='Z:\SEED', win_duration=3, sessions=[0, 1, 2]):

        self.sess = sessions

        data_dir = os.path.join(path, 'Preprocessed_EEG')

        if not os.path.exists(data_dir):
            raise (FileNotFoundError(["Error SEED Dataset: ", path]))

        self._data_paths = []
        data_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

        if not data_paths:
            raise (FileNotFoundError(path))

        for data_path in data_paths:
            if os.name == "nt":
                self._data_paths.append('file://' + data_path.replace('\\', '/'))
            else:
                self._data_paths.append('file://' + data_path)

        self.experiment_name = 'SEED'

        self.paradigm = 'emotion'

        self.srate = 200

        self._START = ['0:06:13',
                 '0:00:50',
                 '0:20:10',
                 '0:49:58',
                 '0:10:40',
                 '1:05:10',
                 '2:01:21',
                 '2:59',
                 '1:18:57',
                 '11:32',
                 '10:41',
                 '2:16:37',
                 '5:36',
                 '35:00',
                 '1:48:53']

        self._END = ['0:10:11',
               '0:04:36',
               '0:23:35',
               '0:54:00',
               '0:13:44',
               '1:08:29',
               '2:05:21',
               '6:40',
               '1:23:23',
               '15:33',
               '14:41',
               '2:20:37',
               '9:36',
               '39:02',
               '1:52:18']

        self._SEQUENCE = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

        self.subjects = list(range(1, 16))

        self.duration = win_duration
        self.win_len = self.duration * self.srate

        self.video_seconds = []
        for i in range(len(self._START)):
            self.video_seconds.append(calculate_duration(self._START[i], self._END[i]))

        if self.duration > min(self.video_seconds):
            raise (ValueError(["Select duration is too long, maximum: ", min(self.video_seconds)]))

        self.events_name = ['sad', 'neutral', 'happy']

        self._CHANNELS = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
                            "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
                            "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
                            "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
                            "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
                            "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "POO7", "O1", "OZ",
                            "O2", "POO8"]


        self._EVENTS = {event_name:
                            (i + 1, (0, self.duration))
                        for i, event_name in enumerate(self.events_name)}



        super().__init__(
            dataset_code=self.experiment_name,
            subjects=self.subjects,
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=self.srate,
            paradigm=self.paradigm,
        )


    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:

        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        subject = int(subject)

        self._data_paths.sort(reverse=False)

        URL = []
        for data_path in self._data_paths:
            if data_path.split('/')[-1].split('_')[0] == str(subject):
                URL.append(data_path)

        if path == None:
            mne_home = os.path.expanduser('~')
            mne_dir = os.path.join(mne_home, 'AssistBCI\\mne_Raw_da')
            if not os.path.exists(mne_dir):
                os.makedirs(mne_dir)

        file_dest = []
        for url in URL:
            file_dest.append(mne_data_path(
                url,
                self.experiment_name,
                path=mne_dir,
                proxies=proxies,
                force_update=force_update,
                update_path=False,
            ))

        return file_dest

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:

        _dests = self.data_path(subject)

        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        ch_names = [ch_name.upper() for ch_name in self._CHANNELS]

        ch_names = ch_names + ["STI 014"]

        ch_types = ["eeg"] * (len(self._CHANNELS) + 1)
        ch_types[-1] = "stim"


        info = create_info(ch_names=ch_names,
                           ch_types=ch_types, sfreq=self.srate)

        # 不同 session 的循环函数
        sess = {}
        for s, dests in enumerate(_dests):

            if s not in self.sess:
                continue

            try:
                raw_mat = scipy.io.loadmat(dests)
            except:
                print("Error loading file: ", dests)
                continue

            # xxx_eeg1 ~ 15
            # 不同 class 的循环函数
            runs = dict()
            for block in range(len(self._SEQUENCE)):

                label = self._SEQUENCE[block] + 1
                block = block + 1

                key = self.search(raw_mat, '_eeg' + str(block))
                data = raw_mat[key]
                data = np.append(data, np.zeros((1, data.shape[1])), axis=0)

                trigger_ind = [0 + self.win_len * i for i in range(data.shape[1] // self.win_len)]
                data[-1, trigger_ind] = label

                raw = RawArray(
                    data=np.reshape(data, (data.shape[0], -1)),
                    info=info
                )
                raw.set_montage(montage)
                runs["run_{:d}".format(block)] = raw

            sess["session_{:d}".format(s)] = runs

        return sess

    def search(self, myDict, lookup):
        for key, value in myDict.items():
            if str.find(key, lookup) != -1:
                return key
