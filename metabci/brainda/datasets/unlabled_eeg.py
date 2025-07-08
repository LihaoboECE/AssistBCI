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
import pyedflib
import scipy.io
from scipy.signal import decimate

'''
read unlabeled eeg from edf file
for emotion or physiological status eeg data
- used in un-supervised learning
Autor: Li Haobo
Email: lihaoboece@gmail.com
2025/7/8
'''


class unLabeled_EEG(BaseDataset):
    '''
    data_path: edf file data path
    win_duration: time duration for one segment
    drate: the sampling rate after down-sampled
    D: downsampling ratio
    '''

    def __init__(self, data_paths=[], win_duration=3, drate=200, D=5):
        for path in data_paths:
            if not os.path.exists(path):
                raise (FileNotFoundError(["Error Dataset: ", path]))

        self.duration = win_duration

        self._data_paths = []
        for data_path in data_paths:
            if os.name == "nt":
                self._data_paths.append('file://' + data_path.replace('\\', '/'))
            else:
                self._data_paths.append('file://' + data_path)

        self.experiment_name = 'SelfSupervised'

        self.paradigm = 'emotion'

        self.D = D

        self._CHANNELS = ["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]

        self._EVENTS = {'unknown': (1, (0, self.duration))}

        super().__init__(
            dataset_code=self.experiment_name,
            subjects=[1],
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=drate,
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

        self._data_paths.sort(reverse=False)

        if path == None:
            mne_home = os.path.expanduser('~')
            mne_dir = os.path.join(mne_home, 'AssistBCI\\mne_Raw_da')
            if not os.path.exists(mne_dir):
                os.makedirs(mne_dir)

        file_dest = []
        for url in self._data_paths:
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

        sess = {}
        for k, file_path in enumerate(_dests):

            try:
                # 打开EDF文件
                edf_file = pyedflib.EdfReader(file_path)

                # 获取文件基本信息
                num_channels = edf_file.signals_in_file
                channel_names = self._CHANNELS
                sample_frequencies = [edf_file.getSampleFrequency(i) for i in range(num_channels)]
                file_duration = edf_file.file_duration

                print(f"EDF文件基本信息:")
                print(f"- 通道数量: {num_channels}")
                print(f"- 文件时长: {file_duration:.2f} 秒")
                print(f"- 通道名称: {channel_names}")
                print(f"- 采样频率: {sample_frequencies} Hz")

                # 读取所有通道的信号数据
                signals = []
                for i in range(num_channels):
                    eeg = edf_file.readSignal(i)
                    signals.append(eeg)

                signals = np.stack(signals)

                # 关闭EDF文件
                edf_file.close()

            except Exception as e:
                print(f"读取EDF文件时出错: {str(e)}")
                return None

            signals = decimate(signals, self.D)
            keep_len = signals.shape[-1] % int(self.duration * self.srate)
            signals = signals[:, :-keep_len]

            trigger = np.zeros((signals.shape[-1]))
            trigger[:: int(self.duration * self.srate)] = 1

            signals = np.concatenate((signals, trigger[np.newaxis,...]), axis=0)

            raw = RawArray(
                data=np.reshape(signals, (signals.shape[0], -1)),
                info=info
            )

            raw.set_montage(montage)
            sess["session_{:d}".format(k)] = {"run_{:d}".format(1): raw}

        return sess
