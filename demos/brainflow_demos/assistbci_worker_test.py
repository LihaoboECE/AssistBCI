# -*- coding: utf-8 -*-
import time
from metabci.brainflow.amplifiers import Marker, NeuroDance
from demos.brainstim_demos.workers import ConditionWorker

'''
A simple start up for assistBCI-v2025 worker
Author: Li Haobo
Email: lihaoboece@gmail.com
#assistBCI-v2025
'''


if __name__ == '__main__':
    srate = 1000

    stim_interval = [0,5]

    feedback_worker_name = 'feedback_worker'

    worker = ConditionWorker(
        timeout=5e-2,
        worker_name=feedback_worker_name,
        srate=srate)
    marker = Marker(interval=stim_interval, srate=srate)

    ns = NeuroDance(##10.8.52.19
        device_address=('127.0.0.1', 8899),
        srate=srate,
        num_chans=8)

    ns.connect_tcp()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(2)
    input('press any key to start\n')

    ns.start_trans()

    input('press any key to close\n')
    ns.down_worker('feedback_worker')
    time.sleep(1)

    ns.stop_trans()
    ns.close_connection()
    print('here')
    ns.clear()
    print('bye')
