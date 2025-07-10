import mmap
import os
import struct
import json
import threading
import tempfile
from collections.abc import MutableMapping

'''
new version for assistbci shared memory,
easy tool for system data/flags storage and transmission
aimed to replace "multiprocessing.Manager().dict()"

Using memory map (Mmap) technic, 
and enable to automatically expand memory during usage

Author: Lihaobo
Email: lihaoboece@gmail.com
2025.7.2
#assistBCI-v2025
'''

# 使用系统临时目录创建共享文件
SHARED_MEM_FILE = os.path.join(tempfile.gettempdir(), 'assistbci_shared.bin')
# 初始共享内存大小 (1MB)
INITIAL_SIZE = 1024 * 1024
# 内存不足时的增长因子
GROWTH_FACTOR = 2
# 文件头大小 (4字节长度 + 4字节容量 + 4字节锁状态)
HEADER_SIZE = 12


class SharedDict(MutableMapping):
    def __init__(self, initial_size=INITIAL_SIZE):
        self.mem_file = SHARED_MEM_FILE
        self.lock = threading.Lock()
        self.initial_size = initial_size
        self._ensure_shared_memory()
        self._map = self._create_mmap()

    def _ensure_shared_memory(self):
        """创建或调整共享内存文件"""
        with self.lock:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.mem_file), exist_ok=True)

            # 如果文件不存在则创建
            if not os.path.exists(self.mem_file):
                with open(self.mem_file, 'wb') as f:
                    # 写入初始头信息和空数据区
                    f.write(struct.pack('III', 0, self.initial_size - HEADER_SIZE, 0))
                    f.write(b'\x00' * (self.initial_size - HEADER_SIZE))

            # 检查文件大小是否足够
            file_size = os.path.getsize(self.mem_file)
            if file_size < HEADER_SIZE:
                with open(self.mem_file, 'wb') as f:
                    f.write(struct.pack('III', 0, self.initial_size - HEADER_SIZE, 0))
                    f.write(b'\x00' * (self.initial_size - HEADER_SIZE))

    def _create_mmap(self):
        """创建内存映射"""
        with open(self.mem_file, 'r+b') as f:
            size = os.path.getsize(self.mem_file)
            return mmap.mmap(f.fileno(), size, access=mmap.ACCESS_WRITE)

    def _resize_memory(self, new_size):
        """调整共享内存大小"""
        with self.lock:
            # 保存当前数据
            if hasattr(self, '_map') and self._map:
                used_size = self._get_used_size()
                current_data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
                self._map.close()
            else:
                current_data = b''

            # 调整文件大小
            with open(self.mem_file, 'r+b') as f:
                f.truncate(new_size)

            # 重新创建内存映射
            with open(self.mem_file, 'r+b') as f:
                self._map = mmap.mmap(f.fileno(), new_size, access=mmap.ACCESS_WRITE)

            # 恢复数据并更新头信息
            if current_data:
                self._map[HEADER_SIZE:HEADER_SIZE + len(current_data)] = current_data

            # 设置头信息：已用大小和数据区容量（不包括头）
            self._set_header(len(current_data), new_size - HEADER_SIZE)

    def _get_header(self):
        """读取头信息"""
        used_size = struct.unpack('I', self._map[:4])[0]
        data_capacity = struct.unpack('I', self._map[4:8])[0]
        lock_state = struct.unpack('I', self._map[8:12])[0]
        return used_size, data_capacity, lock_state

    def _set_header(self, used_size, data_capacity):
        """设置头信息"""
        self._map[:4] = struct.pack('I', used_size)
        self._map[4:8] = struct.pack('I', data_capacity)
        self._map[8:12] = struct.pack('I', 0)

    def _get_used_size(self):
        """获取已使用空间大小"""
        return struct.unpack('I', self._map[:4])[0]

    def _set_used_size(self, size):
        """设置已使用空间大小"""
        self._map[:4] = struct.pack('I', size)

    def _get_data_capacity(self):
        """获取数据区容量（不包括头）"""
        return struct.unpack('I', self._map[4:8])[0]

    def _acquire_lock(self):
        """获取共享锁"""
        while True:
            current_lock = struct.unpack('I', self._map[8:12])[0]
            if current_lock == 0:
                self._map[8:12] = struct.pack('I', 1)
                return
            threading.Event().wait(0.001)

    def _release_lock(self):
        """释放共享锁"""
        self._map[8:12] = struct.pack('I', 0)

    def _check_capacity(self, required):
        """检查并调整容量"""
        used_size = self._get_used_size()
        data_capacity = self._get_data_capacity()
        if used_size + required > data_capacity:
            new_data_capacity = max(self.initial_size - HEADER_SIZE,
                                    int((used_size + required) * GROWTH_FACTOR))
            new_file_size = HEADER_SIZE + new_data_capacity
            self._resize_memory(new_file_size)

    def __setitem__(self, key, value):
        """设置键值对"""
        if not isinstance(key, str):
            raise TypeError("Key must be a string")

        data = json.dumps({key: value}).encode('utf-8')
        data_size = len(data)

        self._acquire_lock()
        try:
            self._check_capacity(data_size)

            used_size = self._get_used_size()
            current_data = self._map[HEADER_SIZE:HEADER_SIZE + used_size] if used_size > 0 else b''

            if current_data:
                current_dict = json.loads(current_data.decode('utf-8'))
            else:
                current_dict = {}

            current_dict[key] = value
            new_data = json.dumps(current_dict).encode('utf-8')
            new_size = len(new_data)

            self._map[HEADER_SIZE:HEADER_SIZE + new_size] = new_data
            self._set_used_size(new_size)
        finally:
            self._release_lock()

    def __getitem__(self, key):
        """获取值"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                raise KeyError(key)

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            data_dict = json.loads(data.decode('utf-8'))
            if key not in data_dict:
                raise KeyError(key)
            return data_dict[key]

    def __delitem__(self, key):
        """删除键"""
        self._acquire_lock()
        try:
            used_size = self._get_used_size()
            if used_size == 0:
                raise KeyError(key)

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            data_dict = json.loads(data.decode('utf-8'))

            if key not in data_dict:
                raise KeyError(key)

            del data_dict[key]
            new_data = json.dumps(data_dict).encode('utf-8')
            new_size = len(new_data)

            self._map[HEADER_SIZE:HEADER_SIZE + new_size] = new_data
            self._set_used_size(new_size)
        finally:
            self._release_lock()

    def __iter__(self):
        """迭代键"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                return iter({})

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            return iter(json.loads(data.decode('utf-8')))

    def __len__(self):
        """获取键数量"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                return 0

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            return len(json.loads(data.decode('utf-8')))

    def __contains__(self, key):
        """检查键是否存在"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                return False

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            data_dict = json.loads(data.decode('utf-8'))
            return key in data_dict

    def clear(self):
        """清空字典"""
        self._acquire_lock()
        try:
            self._set_used_size(0)
            # 获取当前数据区容量
            data_capacity = self._get_data_capacity()
            # 确保我们不会尝试写入超出映射范围
            if data_capacity > 0:
                # 计算实际可写入的数据区大小
                actual_data_size = len(self._map) - HEADER_SIZE
                write_size = min(data_capacity, actual_data_size)
                if write_size > 0:
                    self._map[HEADER_SIZE:HEADER_SIZE + write_size] = b'\x00' * write_size
        finally:
            self._release_lock()

    def keys(self):
        """获取所有键"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                return []

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            return list(json.loads(data.decode('utf-8')).keys())

    def values(self):
        """获取所有值"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                return []

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            return list(json.loads(data.decode('utf-8')).values())

    def items(self):
        """获取所有键值对"""
        with self.lock:
            used_size = self._get_used_size()
            if used_size == 0:
                return []

            data = self._map[HEADER_SIZE:HEADER_SIZE + used_size]
            return list(json.loads(data.decode('utf-8')).items())

    def get(self, key, default=None):
        """安全获取值"""
        try:
            return self[key]
        except KeyError:
            return default

    def __del__(self):
        """清理资源"""
        if hasattr(self, '_map') and self._map:
            self._map.close()


# 使用示例
if __name__ == "__main__":
    # 程序1 - 写入数据
    d1 = SharedDict()
    d1['app'] = 'Shared Dictionary'
    d1['version'] = 1.0
    d1['config'] = ['max_connections', 'timeout']

    print("Program 1 wrote data")

    # 程序2 - 读取数据
    d2 = SharedDict()
    print("\nProgram 2 reading data:")
    print("Keys in shared dict:", list(d2.keys()))
    print("Value of 'app':", d2['app'])
    print("Value of 'config':", d2['config'])

    # 程序3 - 更新数据
    d3 = SharedDict()
    d3['version'] = 2.0
    del d3['config']
    d3['features'] = ['dynamic memory', 'concurrency safe']

    print("\nProgram 3 updated data")

    # 验证更新
    print("\nAll programs can see updated data:")
    print("Keys in shared dict:", list(d1.keys()))
    print("Version:", d1['version'])
    print("Features:", d1['features'])

    # 清空字典
    d1.clear()
    print("\nAfter clear:", len(d1), "items")
    print("File location:", SHARED_MEM_FILE)