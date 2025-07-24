from metabci.brainstim.utils import Light_trigger, Virtual_trigger
import time
from pylsl.pylsl import StreamInlet, resolve_byprop
from metabci.utils.sharedmemory import SharedDict

if __name__ == "__main__":

    #光电管测试
    LT = Light_trigger(lsl_source_id="trigger", w=1920, h=1080)
    LT.start()
    LT.setData(0) #启动打标
    time.sleep(1)

    streams = resolve_byprop("source_id", "trigger", timeout=1)
    inlet = StreamInlet(streams[0])
    inlet.pull_sample(timeout=0)

    time.sleep(1)

    # 打标
    for i in range(10):
        time.sleep(0.5)
        LT.setData(i+1)
        samples, timestamp = inlet.pull_sample(timeout=1)
        print("Event: ", samples, "Timestamp: ", timestamp)
        #此处在设备中进行光电trigger event的替换

    # 安全退出
    LT.setData(-1)


    #虚拟打标测试
    SD = SharedDict() #基于Mmap的共享内存， 可同时开启metabci/utils/sharedmemory_ManageTool.py进行查看

    print("port 0 test")
    VT = Virtual_trigger(port=0, dict=SD)
    for i in range(5):
        time.sleep(0.5)
        VT.setData(i+1)
        print("trigger: ", SD["Virtual_trigger"])

    print("port 1 test")
    VT = Virtual_trigger(port=1, dict=SD)
    for i in range(5):
        time.sleep(0.5)
        VT.setData(i+1)
        print("trigger: ", SD["Virtual_trigger"])

    print("port 2 test")
    VT = Virtual_trigger(port=2, dict=SD)
    for i in range(5):
        time.sleep(0.5)
        VT.setData(i + 1)
        print("trigger: ", SD["Virtual_trigger"])




