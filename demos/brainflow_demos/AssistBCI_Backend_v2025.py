from demos.brainflow_demos.device_worker import Device

'''
Start assistBCI-v2025 device worker backend
'''

if __name__ == "__main__":
    device = Device()
    device.start()
    device.join()



