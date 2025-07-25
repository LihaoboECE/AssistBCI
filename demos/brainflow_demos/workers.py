from metabci.brainflow.workers import ProcessWorker
import numpy as np
import os
from scipy.io import loadmat, savemat
from scipy import signal
from metabci.utils.sharedmemory import SharedDict
from metabci.brainda.algorithms.self_supervised_learning.prototype import PCA_EmoAdapt_online
import time
import torch

'''
workers used in assistBCI-v2025

Author: Li Haobo
Email: lihaoboece@gmail.com
'''

class EmptyWorker(ProcessWorker):
    def __init__(self, timeout, worker_name):
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        print("Entering Pre process")
        pass
    def consume(self, data):
        print("Entering Consume process")
        print(f"With data shape: {np.array(data).shape}")
        pass
    def post(self):
        print("Entering Post process")
        pass


##used for assistbci v2025
class ConditionWorker(ProcessWorker):
    DEFAULT_CHANNEL_NUM = 8
    def __init__(self, timeout, worker_name, srate):
        self.gmm = None
        self.srate = srate
        _buffer = SharedDict()
        _buffer['AG_control'] = 'disable'
        _buffer['AG_state'] = 'disable' #used for broadcast personal state
        _buffer['AG_feedback'] = {} #feedback including predict T/F report & new class report
        #format {timestamp: feedback}

        _buffer['AG_predict'] = {'new': False, 'labels': [None], 'report': False, 'timestamp': int(time.time() * 1000)}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.root = os.path.join('..', '..', 'assistbci_models')
        self.model_path = os.path.join(self.root, 'EmoAdapt', '0', 'model', 'best_model.pth')
        self.classifier_data_path = os.path.join(self.root, 'classifier')

        self.data_buffer = []
        self.label_buffer = []

        self.temp_buffer = []

        self.waiting_buffer_max_len = 20
        self.waiting_buffer = {} #waiting for feedback, user feedback duration is unknown
        # format report: {str: timestamp: (data, label)}
        #       new class: {str: timestamp: (data, None)}

        super().__init__(timeout=timeout, name=worker_name)


    def pre(self):
        self._buffer = SharedDict()

        model = torch.load(self.model_path, map_location='cpu', weights_only=False)['model']

        classifier_data, classifier_labels = [], []
        if os.path.exists(self.classifier_data_path):
            for filename in os.listdir(self.classifier_data_path):
                if filename.endswith('.mat'):
                    _name = os.path.join(self.classifier_data_path, filename)
                    _data = loadmat(_name)
                    classifier_data.append(_data['data'])
                    classifier_labels.append(_data['labels'])
        if len(classifier_data) != 0:
            classifier_data = np.concatenate(classifier_data, axis=0)
            classifier_labels = np.concatenate(classifier_labels, axis=0)
            self.gmm = PCA_EmoAdapt_online(n_features=5, backbone=model, device=self.device,
                                           original_buf_EEG=classifier_data, buf_labels=classifier_labels,
                                           pca_patch_max_len=20, max_buf_size=50,
                                           min_samples_per_class=20, adjustment_step=0.5, max_adjustment=10.0)
            self.gmm.init_model()
            self._buffer['AG_state'] = 'enable'
        else:
            self.gmm = PCA_EmoAdapt_online(n_features=5, backbone=model, device=self.device,
                                          pca_patch_max_len=20, max_buf_size=50,
                                          min_samples_per_class=20, adjustment_step=0.5, max_adjustment=10.0)
            self._buffer['AG_state'] = 'pre_train'

        del classifier_data, classifier_labels
        return


    def consume(self, raw_data):

        if np.array(raw_data).shape[-1] > self.DEFAULT_CHANNEL_NUM + 1:
            print(f"Using First {self.DEFAULT_CHANNEL_NUM} channels, "
                  f"for device more then {self.DEFAULT_CHANNEL_NUM} channels,"
                  f"please switch channels to suitable montage")

        if self._buffer['AG_control'] == "disable":
            return

        START_TIME = time.time()

        x = np.array(raw_data).T[np.newaxis, :8,...]
        x = signal.decimate(x, self.srate//200) #200Hz model

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

        if self._buffer['AG_state'] == 'enable':

            # 参数
            MIN_DETECTION_INTERVAL = 12  # 最小检测间隔（样本数）
            MIN_ACC_NEW_CLASS = 3
            MAX_BUFFER_SIZE = 3  # 最大缓存样本数

            self.gmm.sample_counter += 1

            new, labels, report = self.gmm.predict(x)
            new, labels, report = new[0], labels[0], report[0] # predict for one sample
            # new = True

            if new:
                # 维护固定大小的临时缓存
                self.temp_buffer.append(x)
                if len(self.temp_buffer) > MAX_BUFFER_SIZE:
                    self.temp_buffer.pop(0)

                # 多级检测决策
                if self.gmm.should_trigger_new_class_detection(labels[0], np.concatenate(self.temp_buffer, axis=0),
                                                          MIN_DETECTION_INTERVAL, MIN_ACC_NEW_CLASS):
                    self.gmm.reset_detection_state(labels[0])
                    _current_time = int(time.time() * 1000)
                    buffer_array = np.concatenate(self.temp_buffer, axis=0)
                    self.waiting_buffer[_current_time] = (buffer_array, None)
                    self._buffer['AG_predict'] = {'new': True, 'labels': labels, 'report': False,
                                                  'timestamp': _current_time}
                    print("new")

            else:
                # 重置缓存
                self.temp_buffer = []
                self.gmm.reset_detection_state(labels[0])

                if report:
                    _current_time = int(time.time() * 1000)
                    self.waiting_buffer[_current_time] = (x, labels[0])
                    self._buffer['AG_predict'] = {'new': False, 'labels': labels, 'report': True,
                                                  'timestamp': _current_time}
                    print("report: ", labels)


            if len(self.waiting_buffer) > self.waiting_buffer_max_len:
                _keys = sorted(self.waiting_buffer.keys())[-self.waiting_buffer_max_len:]
                [self.waiting_buffer.pop(_key) for _key in _keys]
                print(self.waiting_buffer)


            if len(self._buffer['AG_feedback']) != 0:
                for timestamp in self._buffer['AG_feedback'].keys():
                    timestamp = int(timestamp)
                    if timestamp in self.waiting_buffer.keys():
                        _feedbacks = self._buffer['AG_feedback']
                        _feedback = _feedbacks.pop(str(timestamp))
                        self._buffer['AG_feedback'] = _feedbacks

                        _data, _label = self.waiting_buffer.pop(timestamp)
                        if type(_feedback) == bool: # predict feedback
                            if _feedback: # correct predict
                                self.data_buffer.append(_data)
                                self.label_buffer.append(_label)
                                print("预测正确反馈")
                            else:
                                self.gmm.error_pred_feedback(_label)
                                print("预测错误反馈")
                        elif type(_feedback) == str: # new class feedback
                            self.data_buffer.append(_data)
                            self.label_buffer.append(np.array([_feedback for _ in range(_data.shape[0])]))
                            print("新状态反馈")
                        else:
                            print("UnKnown feedback")
                    else:
                        print("feedback timeout, buffer is cleared before feedback")
                        _feedbacks = self._buffer['AG_feedback']
                        _feedback = _feedbacks.pop(str(timestamp))
                        self._buffer['AG_feedback'] = _feedbacks
                        continue

                if len(self.data_buffer) >= 1:
                    self.data_buffer = np.concatenate(self.data_buffer, axis=0)
                    try:
                        self.label_buffer = np.concatenate(self.label_buffer, axis=0)
                    except:
                        self.label_buffer = np.array(self.label_buffer)
                else:
                    return

                self.gmm.fit(self.data_buffer, self.label_buffer)

                if time.time() - START_TIME <= 4:
                    _name = os.path.join(self.classifier_data_path, str(int(START_TIME * 1000))+'.mat')
                    savemat(_name, {'data': self.data_buffer, 'labels': self.label_buffer})
                else:
                    print("Quite data saving: Timeout")

                self.data_buffer, self.label_buffer = [], []

        elif self._buffer['AG_state'] == 'pre_train':

            MIN_INIT_SAMPLES = 12
            self.temp_buffer.append(x)

            self._buffer["pre train 进度"] = f"{int((len(self.temp_buffer) / MIN_INIT_SAMPLES) * 100)}%"


            if len(self.temp_buffer) >= MIN_INIT_SAMPLES:
                self.temp_buffer = np.concatenate(self.temp_buffer, axis=0)
                self.gmm.fit(self.temp_buffer, np.array(["中性" for _ in range(self.temp_buffer.shape[0])]))

                if time.time() - START_TIME <= 4:
                    if not os.path.exists(self.classifier_data_path):
                        os.makedirs(self.classifier_data_path)
                    _name = os.path.join(self.classifier_data_path, str(int(START_TIME * 1000)) + '.mat')
                    savemat(_name, {'data': self.temp_buffer, 'labels': np.array(["中性" for _ in range(self.temp_buffer.shape[0])])})
                else:
                    print("Quite data saving: Timeout")

                del self._buffer["pre train 进度"]
                self.temp_buffer = []
                self._buffer['AG_state'] = 'enable'

        if time.time() - START_TIME > 5:
            print("Loss synchronization, Please use a better PC: ", time.time() - START_TIME, "sec")

    def post(self):
        pass