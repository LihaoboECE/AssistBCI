import time
import numpy as np

import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, Marker, BlueBCI
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBTDCA
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin
from multiprocessing import Lock
from scipy.stats import kurtosis
from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.pipeline import clone
from sklearn.metrics import balanced_accuracy_score
import os
import pickle
from metabci.brainda.datasets import Experiment
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices, match_loo_indices)
from metabci.brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP, SCCA, TDCA,
    generate_filterbank, generate_cca_references)
from scipy.stats import zscore
import time





class BasicWorker(ProcessWorker):
    def __init__(self, timeout, worker_name):

        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        print("-------------------Entering Pre-------------------")
        print("start trainning---------------")
        dataset = Experiment()
        delay = 0.3  # seconds
        self.delay = delay

        channels = ['PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
        srate = 1000  # Hz
        duration = 4  # seconds
        self.duration = duration

        n_bands = 5
        n_harmonics = 4
        events = sorted(list(dataset.events.keys()))
        freqs = [dataset.get_freq(event) for event in events]
        phases = [dataset.get_phase(event) for event in events]

        Yf = generate_cca_references(
            freqs, srate, duration,
            phases=None,
            n_harmonics=n_harmonics)

        start_pnt = dataset.events[events[0]][1][0]
        self.start_pnt = start_pnt

        paradigm = SSVEP(
            srate=srate,
            channels=channels,
            intervals=[(start_pnt + delay, start_pnt + delay + duration + 0.1)],  # more seconds for TDCA
            events=events)

        wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
        ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
        filterbank = generate_filterbank(
            wp, ws, srate, order=4, rp=1)
        filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25

        def data_hook(X, y, meta, caches):
            filterbank = generate_filterbank(
                [[8, 90]], [[6, 95]], srate, order=4, rp=1)
            X = sosfiltfilt(filterbank[0], X, axis=-1)
            return X, y, meta, caches

        paradigm.register_data_hook(data_hook)

        set_random_seeds(64)
        l = 5
        models = OrderedDict([
            ('fbscca', FBSCCA(
                filterbank, filterweights=filterweights)),
            # ('fbecca', FBECCA(
            #     filterbank, filterweights=filterweights)),
            # ('fbdsp', FBDSP(
            #     filterbank, filterweights=filterweights)),
            # ('fbtrca', FBTRCA(
            #     filterbank, filterweights=filterweights)),
            # ('fbtdca', FBTDCA(
            #     filterbank, l, n_components=8,
            #     filterweights=filterweights)),
        ])

        X, y, meta = paradigm.get_data(
            dataset,
            subjects=[2],
            return_concat=True,
            n_jobs=1,
            verbose=False)

        set_random_seeds(42)
        loo_indices = generate_loo_indices(meta)

        for model_name in models:
            if model_name == 'fbtdca':
                filterX, filterY = np.copy(X[..., :int(srate * duration) + l]), np.copy(y)
            else:
                filterX, filterY = np.copy(X[..., :int(srate * duration)]), np.copy(y)

            filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

            trainX, trainY = filterX[:], filterY[:]
            self.classfier = clone(models[model_name]).fit(
                    trainX, trainY,
                    Yf=Yf
                )
            print("Ready to start---------------")

    def consume(self, data):
        print("-------------------Entering consume-------------------")
        # if data:
        #     print("data:", data)
        print("inside consume")

        data = np.array(data)

        srate = 1000  # Hz
        interval = [(self.start_pnt + self.delay)*srate, (self.start_pnt + self.delay + self.duration + 0.1)*srate]
        data = np.transpose(data, [1, 0])
        data = data[:-1,:]
        filterbank = generate_filterbank(
            [[8, 90]], [[6, 95]], srate, order=4, rp=1)
        data = sosfiltfilt(filterbank[0], data, axis=-1)
        data = data[np.newaxis, :, int(interval[0]):int(interval[1])]
        data = data[..., :int(srate * self.duration)]
        print("here1")
        print("data size:", np.size(data))
        p_labels = self.classfier.predict(data)
        #p_labels = [1]
        print("here2")
        p_labels = int(p_labels[0]) + 1

        self.controller.CMD(self.CMD_list[p_labels-1])
        print("CMD:", self.CMD_list[p_labels-1])
        print("moved")

        print('predict_id_paradigm', p_labels)


    def post(self):
        pass


class ControlWorker(ProcessWorker):
    def __init__(self, timeout, worker_name, dict):
        self.lock = Lock()
        self._buffer = dict
        super().__init__(timeout=timeout, name=worker_name)

        self.use_model = True

    def send(self, name, data):
        self.lock.acquire()
        try:
            self._buffer[name] = data
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def pre(self):

        self.send("par_feedback", False)

        # model loading
        home_dir = os.path.join(os.path.expanduser('~'), 'AssistBCI\\Personal_Model')

        try:
            model_paths = [os.path.join(home_dir, file) for file in os.listdir(home_dir)]

            def time_sort(item):
                return int("".join(item.split("\\")[-1].split(".")[0].split("_")[1:]))

            model_path = sorted(model_paths, key=time_sort, reverse=True)
            print(model_path)

            with open(str(model_path[0]), 'rb') as file:
                self.classfier = pickle.load(file)

            self.kurtosis = 2

            print("model loaded")

        except:
            try:
                freqs = np.arange(8, 8 + 1 * 8+0.1, 1).tolist()
                self.duration = 3  # seconds
                self.n_harmonics = 4
                self.srate = 1000
                Yf = generate_cca_references(
                    freqs, self.srate, self.duration,
                    phases=None,
                    n_harmonics=self.n_harmonics)

                self.classfier = clone(SCCA()).fit(Yf=Yf)

                self.kurtosis = 3

                print("General model is used")

            except:
                self.use_model = False
                print("General model also Fail to load")

            return



    def consume(self, raw_data):
        if not self._buffer['CMD_list'] or not self._buffer["current_par"]:
            #par not started
            return

        if self.use_model:
            self.delay = 0  # seconds
            self.duration = 3  # seconds
            self.n_bands = 5
            self.n_harmonics = 4

            data = np.array(raw_data)

            srate = 1000  # Hz
            interval = [(0.14 + self.delay) * srate, (0.14 + self.delay + self.duration) * srate]
            data = np.transpose(data, [1, 0])
            data = data[:-1, :]

            filterbank = generate_filterbank(
                [[8, 90]], [[6, 95]], srate, order=4, rp=1)
            data = sosfiltfilt(filterbank[0], data, axis=-1)

            data = data[np.newaxis, :, int(interval[0]):int(interval[1])]
            data = data[..., :int(srate * self.duration)]


            # print("---CMD---:", self._buffer['CMD_list'][self._buffer["current_par"]])
            try:
                labels, features = self.classfier.predict(data)
                features = features[:len(self._buffer['CMD_list'][self._buffer["current_par"]])]
                if kurtosis(np.sort(features), axis=-1, fisher=False) > self.kurtosis:
                    p_labels = np.argmax(features, axis=-1)
                    p_labels = int(p_labels[0])
                    print("good kurtosis:", kurtosis(np.sort(features), axis=-1, fisher=False))
                    # print('labels:', p_labels)
                    self.send("CMD_label", p_labels)
                else:
                    print("kurtosis:", kurtosis(np.sort(features), axis=-1, fisher=False))

            except:
                print("Error")

            finally:
                self.clear_queue()
                self.send("par_feedback", True)

    def post(self):
        pass


# class ControlWorker(ProcessWorker):
#     def __init__(self, timeout, worker_name, dict):
#         self.lock = Lock()
#         self._buffer = dict
#         super().__init__(timeout=timeout, name=worker_name)
#
#         self.use_model = True
#
#     def send(self, name, data):
#         self.lock.acquire()
#         try:
#             self._buffer[name] = data
#         finally:
#             # 无论如何都要释放锁
#             self.lock.release()
#
#     def pre(self):
#
#         self.send("par_feedback", False)
#
#         # model loading
#         home_dir = os.path.join(os.path.expanduser('~'), 'AssistBCI\\Personal_Model')
#
#         try:
#             model_paths = [os.path.join(home_dir, file) for file in os.listdir(home_dir)]
#
#             def time_sort(item):
#                 return int("".join(item.split("\\")[-1].split(".")[0].split("_")[1:]))
#
#             model_path = sorted(model_paths, key=time_sort, reverse=True)
#             print(model_path)
#
#             with open(str(model_path[0]), 'rb') as file:
#                 self.classfier = pickle.load(file)
#
#             self.kurtosis = 3.5
#
#             print("model loaded")
#
#         except:
#             try:
#                 # freqs = np.arange(8, 8 + 1 * 8+0.1, 1).tolist()
#                 # self.duration = 3  # seconds
#                 # self.n_harmonics = 4
#                 # self.srate = 1000
#                 # Yf = generate_cca_references(
#                 #     freqs, self.srate, self.duration,
#                 #     phases=None,
#                 #     n_harmonics=self.n_harmonics)
#                 #
#                 # self.classfier = clone(SCCA()).fit(Yf=Yf)
#
#                 self.kurtosis = 2
#
#                 print("General model will be used")
#
#             except:
#                 self.use_model = False
#                 print("General model also Fail to load")
#
#             return
#
#
#
#     def consume(self, raw_data):
#         if not self._buffer['CMD_list'] or not self._buffer["current_par"]:
#             #par not started
#             return
#
#         if self.use_model:
#             self.delay = 0  # seconds
#             self.duration = 3  # seconds
#             self.n_bands = 5
#             self.n_harmonics = 4
#
#             # raw_data = np.array(raw_data)
#             # try:
#             #     self.data_buff = np.concatenate((self.data_buff, raw_data), axis=0)
#             # except:
#             #     self.data_buff = raw_data
#             #
#             # if self.data_buff.shape[0] > 2000:
#             #     self.data_buff = self.data_buff[-2000:, :]
#             #     data = self.data_buff
#             # else:
#             #     return
#
#             # srate = 1000  # Hz
#             # interval = [self.delay*srate, (self.delay + self.duration)*srate]
#             # data = np.transpose(data, [1, 0])
#             # data = data[:-1,:]
#             # filterbank = generate_filterbank(
#             #     [[8, 90]], [[6, 95]], srate, order=4, rp=1)
#             # data = sosfiltfilt(filterbank[0], data, axis=-1)
#             # data = data[np.newaxis, :, int(interval[0]):int(interval[1])]
#             # data = data[..., :int(srate * self.duration)]
#
#             data = np.array(raw_data)
#
#             srate = 1000  # Hz
#             interval = [(0.14 + self.delay) * srate, (0.14 + self.delay + self.duration) * srate]
#             data = np.transpose(data, [1, 0])
#             data = data[:-1, :]
#
#             filterbank = generate_filterbank(
#                 [[8, 90]], [[6, 95]], srate, order=4, rp=1)
#             data = sosfiltfilt(filterbank[0], data, axis=-1)
#
#             data = data[np.newaxis, :, int(interval[0]):int(interval[1])]
#             data = data[..., :int(srate * self.duration)]
#
#
#             # print("---CMD---:", self._buffer['CMD_list'][self._buffer["current_par"]])
#             try:
#                 self.classfier
#             except:
#                 try:
#                     print("fitting")
#                     freqs = np.arange(8, 8 + 1 * 8 + 0.1, 1).tolist()
#                     self.duration = 3  # seconds
#                     self.n_harmonics = 4
#                     self.srate = 1000
#                     n_bands = 5
#                     wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
#                     ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
#                     filterbank = generate_filterbank(
#                         wp, ws, srate, order=4, rp=1)
#                     filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25
#
#                     Yf = generate_cca_references(
#                         freqs, self.srate, self.duration,
#                         phases=None,
#                         n_harmonics=self.n_harmonics)
#
#                     self.classfier = clone(FBSCCA(filterbank, filterweights=filterweights)).fit(data, Yf=Yf)
#
#                 except:
#                     self.use_model = False
#                     print("General model also Fail to load")
#                     return
#
#             try:
#                 labels, features = self.classfier.predict(data)
#                 features = features[:len(self._buffer['CMD_list'][self._buffer["current_par"]])]
#                 if kurtosis(np.sort(features), axis=-1, fisher=False) > self.kurtosis:
#                     print(features)
#                     p_labels = np.argmax(features, axis=-1)
#                     print(p_labels)
#
#                     p_labels = int(p_labels[0])
#                     # print('labels:', p_labels)
#                     self.send("CMD_label", p_labels)
#
#                 else:
#                     print("kurtosis:", kurtosis(np.sort(features), axis=-1, fisher=False))
#
#             except:
#                 print("Error")
#
#             finally:
#                 self.clear_queue()
#                 self.send("par_feedback", True)
#
#     def post(self):
#         pass



class OnlineTestWorker(ProcessWorker):
    def __init__(self, timeout, worker_name):
        super().__init__(timeout=timeout, name=worker_name)

        self.use_model = True

    def pre(self):

        # model loading
        home_dir = os.path.join(os.path.expanduser('~'), 'AssistBCI\\Personal_Model')

        try:
            model_paths = [os.path.join(home_dir, file) for file in os.listdir(home_dir)]

            def time_sort(item):
                return int("".join(item.split("\\")[-1].split(".")[0].split("_")[1:]))

            model_path = sorted(model_paths, key=time_sort, reverse=True)
            print(model_path)

            with open(str(model_path[0]), 'rb') as file:
                self.classfier = pickle.load(file)

            print("model loaded")

        except:
            try:
                freqs = np.arange(8, 8 + 1 * 10, 1).tolist()
                self.duration = 2  # seconds
                self.n_harmonics = 4
                self.srate = 1000
                Yf = generate_cca_references(
                    freqs, self.srate, self.duration,
                    phases=None,
                    n_harmonics=self.n_harmonics)

                self.classfier = clone(SCCA()).fit(Yf=Yf)

                print("General model is used")

            except:
                self.use_model = False
                print("General model also Fail to load")

        self.lsl_source_id = 'meta_online_worker'
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)

        print('Waiting connection brainstim...')


    def consume(self, raw_data):

        if self.use_model:
            self.delay = 0  # seconds
            self.duration = 2  # seconds
            self.n_bands = 5
            self.n_harmonics = 4

            data = np.array(raw_data)

            srate = 1000  # Hz
            interval = [(0.14 + self.delay)*srate, (0.14 + self.delay + self.duration)*srate]
            data = np.transpose(data, [1, 0])
            data = data[:-1,:]

            filterbank = generate_filterbank(
                [[8, 90]], [[6, 95]], srate, order=4, rp=1)
            data = sosfiltfilt(filterbank[0], data, axis=-1)

            data = data[np.newaxis, :, int(interval[0]):int(interval[1])]
            data = data[..., :int(srate * self.duration)]

            # print("---CMD---:", self._buffer['CMD_list'][self._buffer["current_par"]])

            p_labels, features = self.classfier.predict(data)
            print('predict_id_paradigm', p_labels)
            while not self.outlet.have_consumers():
                time.sleep(0.1)
            self.outlet.push_sample([int(p_labels)+1])
            print("predict label pushed")


    def post(self):
        pass



class EmptyWorker(ProcessWorker):
    def __init__(self, timeout, worker_name):
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        pass
    def consume(self, data):
        pass
    def post(self):
        pass