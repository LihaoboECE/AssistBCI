# -*- coding: utf-8 -*-
"""
Emotion/ physiological status Experiment
Author: Li Haobo
Email: lihaoboece@gmail.com
"""

import time
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import Marker, NeuroDance
from metabci.brainflow.workers import ProcessWorker
from metabci.utils.sharedmemory import SharedDict
import numpy as np

class FeedbackWorker(ProcessWorker):
    def __init__(self, lsl_source_id, timeout, worker_name):
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        print("-------------------Entering Pre-------------------")
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection brainstim...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                # if self.outlet.have_consumers():
                # #不停寻找同lsl_source_id的刺激程序
                break
        print('Connected to brainstim')

    def consume(self, data):
        print("-------------------Entering consume-------------------")

        if data:
            data = np.array(data)
            print(data.shape)
            if (data[:,-1] != 0).any():
                print("tigger: ", data[data[:,-1] != 0])

        p_labels = 1
        print('return fake predict id', p_labels)

        # while not self.outlet.have_consumers():
        #     time.sleep(0.1)
        self.outlet.push_sample([p_labels])
        print("predict label pushed")

    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0, 5]
    # Label types
    stim_labels = list(range(1, 6)) #1,2,3,4,5

    lsl_source_id = 'emotion_experiment_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels,
                    save_data=True)
    #save_data 控制marker是否保存数据， 最终保存时需调用：marker.save_as_mat()

    # Set NeuroDance parameters
    dict = SharedDict()

    dv = NeuroDance(device_address=("127.0.0.1", 8899),
                    srate=srate,
                    dict=dict) #用于Virtual_trigger

    # Start tcp connection with ns
    dv.connect_tcp()

    # Register worker for online data processing
    dv.register_worker(feedback_worker_name, worker, marker)

    # Start online data processing
    dv.up_worker(feedback_worker_name)
    time.sleep(5) #留时间做pre()

    # Start slicing data and passing data to worker
    dv.start_trans()

    input('press any key to close\n')

    # marker.save_as_mat() #保存数据

    dv.down_worker('feedback_worker')
    time.sleep(1)

    # Stop online data retriving of ns
    dv.stop_trans()
    dv.close_connection()
    print('bye')
