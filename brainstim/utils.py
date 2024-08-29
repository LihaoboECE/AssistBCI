import serial
from psychopy import parallel
import numpy as np
from demos.brainstim_demos.app_test import APP
from pylsl import StreamInfo, StreamOutlet
from multiprocessing import Event, Process, Lock
import pyglet
import time
import tkinter as tk
from tkinter import ttk


class NeuroScanPort:
    """
    Send tag communication Using parallel port or serial port.

    author: Lichao Xu

    Created on: 2020-07-30

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.

    Attributes
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.
        port:
            Send tag communication Using parallel port or serial port.

    Tip
    ----
    .. code-block:: python
       :caption: An example of using port to send tags

        from brainstim.utils import NeuroScanPort
        port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
        VSObject.win.callOnFlip(port.setData, 1)
        port.setData(0)

    """

    def __init__(self, port_addr, use_serial=False, baudrate=115200):
        self.use_serial = use_serial
        if use_serial:
            self.port = serial.Serial(port=port_addr, baudrate=baudrate)
            self.port.write([0])
        else:
            self.port = parallel.ParallelPort(address=port_addr)

    def setData(self, label):
        """Send event labels

        Parameters
        ----------
            label:
                The label sent.

        """
        if self.use_serial:
            self.port.write([int(label)])
        else:
            self.port.setData(int(label))


class NeuraclePort:
    """
    Send trigger to Neuracle device.The Neuracle device uses serial
    port for writing trigger, so it does not need to write a 0 trigger
    before a int trigger. This class is writen under the Trigger box instruction.

    author: Jie Mei

    Created on: 2022-12-05

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        baudrate: int
            The serial port baud rate.

    """

    def __init__(self, port_addr, baudrate=115200) -> None:
        # The only choice for neuracle is using serial for writting trigger
        self.port = serial.Serial(port=port_addr, baudrate=baudrate)

    def setData(self, label):
        # Neuracle doesn't need 0 trigger before a int trigger.
        if str(label) != '0':
            head_string = '01E10100'
            hex_label = str(hex(label))
            if len(hex_label) == 3:
                hex_value = hex_label[2]
                hex_label = '0'+hex_value.upper()
            else:
                hex_label = hex_label[2:].upper()
            send_string = head_string+hex_label
            send_string_byte = [int(send_string[i:i+2], 16)
                                for i in range(0, len(send_string), 2)]
            self.port.write(send_string_byte)


class LsLPort:
    """
    Creating a lab streaming layer marker, which could align with the
    stream which retriving stream from devices.

    """

    def __init__(self) -> None:
        self.info = StreamInfo(
            name='LSLMarkerStream',
            type='Marker',
            channel_count=1,
            nominal_srate=0,
            channel_format='cf_int16')
        self.outlet = StreamOutlet(self.info)

    def setData(self, label):
        # We don't need 0 trigger before a int trigger
        if str(label) != '0':
            self.outlet.push_sample(str(label))



class Virtual_trigger:
    def __init__(self, dict, timeout: float = 5):
        self.lock = Lock()
        self._buffer = dict
        self.send("Virtual_trigger", 1) #开启
        #Virtual_trigger 有 0：关闭 1：未激发 2：激发 三种状态

    def send(self, name, data):
        self.lock.acquire()
        try:
            self._buffer[name] = data
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def get(self, name):
        return self._buffer[name]

    def setData(self, label):
        if label:
            self.send("Virtual_trigger", 2) #激发





class Light_trigger(Process):

    def __init__(self, lsl_source_id="trigger", w=1920, h=1080):
        Process.__init__(self)
        self.trigger_ = Event()
        self._exit = Event()
        self._exit.clear()
        self.trigger_.clear()
        self.outlet = []
        self.start_setData = False
        self.lsl_source_id = lsl_source_id
        self.fps = 60
        self.w = w
        self.h = h
        self.win_start = Event()
        self.win_start.clear()

    def toggle_color(self):
        if self.trigger_.is_set():
            self.canvas.itemconfig(self.square, fill="white")
            self.trigger_.clear()
        else:
            current_color = self.canvas.itemcget(self.square, "fill")
            if current_color == "white":
                self.canvas.itemconfig(self.square, fill="black")

    def run(self):
        self.root = tk.Tk()
        self.root.wm_attributes("-topmost", 1)
        self.root.overrideredirect(True)
        # x = int((self.root.winfo_screenwidth() - label.winfo_reqwidth()) / 2)
        # y = int((self.root.winfo_screenheight() - label.winfo_reqheight()) / 2)
        self.root.geometry("+{}+{}".format(-10, int(self.h* 7/8)))
        self.canvas = tk.Canvas(self.root, width=self.w+20, height=int(self.h/8))
        self.canvas.pack()
        self.square = self.canvas.create_rectangle(0, 0, self.w+20, int(self.h/8), fill="black")
        self.win_start.set()

        while not self._exit.is_set():
            self.toggle_color()
            self.root.update()
            time.sleep(1/self.fps)

        self.root.destroy()

    def setData(self, event):
        if event == -1:
            self._exit.set()
        elif self.start_setData and event != 0:
            self.outlet.push_sample([event])
            self.trigger_.set()
            # while not self.outlet.have_consumers():
            #     time.sleep(1e-6)
            # self.outlet.push_sample([event])
            print("send event succeed", event)
        elif not self.start_setData:
            print("--------------------------port start--------------------------")
            info = StreamInfo(
                name='event_transmitter',
                type='event',
                channel_count=1,
                nominal_srate=0,
                channel_format='int32',
                source_id=self.lsl_source_id)
            self.outlet = StreamOutlet(info)
            print('Waiting for Amplifier...')
            self.start_setData = True
            while self.outlet.have_consumers():
                time.sleep(0.5)






def _check_array_like(value, length=None):
    """
    Check array dimensions.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        value: ndarray,
            The array to check.
        length: int,
            The array dimension.

    """

    flag = isinstance(value, (list, tuple, np.ndarray))
    return flag and (len(value) == length if length is not None else True)


def _clean_dict(old_dict, includes=[]):
    """
    Clear dictionary.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        old_dict: dict,
            The dict to clear.
        includes: list,
            Key-value indexes that need to be preserved.

    """

    names = list(old_dict.keys())
    for name in names:
        if name not in includes:
            old_dict[name] = None
            del old_dict[name]
    return old_dict


