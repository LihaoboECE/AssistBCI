import multiprocessing
import os
from demos.brainstim_demos.device_worker import Device
from sharedmemory import SharedDict

'''
Start assistBCI-v2025 device worker backend
'''

if __name__ == "__main__":
    device = Device()
    device.start()
    device.join()



