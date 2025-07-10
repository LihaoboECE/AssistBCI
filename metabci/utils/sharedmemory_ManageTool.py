import sys
import json
import os
import tempfile
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton, QLineEdit,
                             QLabel, QComboBox, QGroupBox, QHeaderView, QMessageBox,
                             QMenu, QAction, QDialog, QTextEdit, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import mmap
import struct
import threading
from collections.abc import MutableMapping

'''
Shared Memory management tool
Be able to visual & editing the shared memory

aimed to replace "multiprocessing.Manager().dict()"
Author: Lihaobo
Email: lihaoboece@gmail.com
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


class SharedMemoryViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.shared_dict = SharedDict()
        self.mem_file = SHARED_MEM_FILE
        self.init_ui()
        self.load_data()

        # 设置自动刷新定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_data)
        self.timer.start(1000)  # 每秒刷新一次

        # 存储当前选中的键
        self.current_selected_key = None

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("开发者工具")
        self.setGeometry(100, 100, 1000, 700)

        # 设置窗口置顶
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 添加标题
        title_label = QLabel("共享内存管理工具")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; background-color: #ecf0f1; padding: 10px;")
        main_layout.addWidget(title_label)

        # 添加内存信息面板
        info_group = QGroupBox("内存信息")
        info_layout = QHBoxLayout()

        self.file_label = QLabel(f"文件: {self.mem_file}")
        self.size_label = QLabel("大小: 计算中...")
        self.items_label = QLabel("项目: 0")
        self.status_label = QLabel("状态: 已连接")

        info_layout.addWidget(self.file_label)
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.items_label)
        info_layout.addWidget(self.status_label)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # 添加表格显示内存内容
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["键", "值", "类型"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # 启用表格右键菜单
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)

        # 连接选择变化信号
        self.table.itemSelectionChanged.connect(self.handle_selection_changed)

        main_layout.addWidget(self.table)

        # 添加操作面板
        operation_group = QGroupBox("操作")
        operation_layout = QVBoxLayout()

        # 添加键值对输入
        input_layout = QHBoxLayout()

        key_layout = QVBoxLayout()
        key_layout.addWidget(QLabel("键:"))
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("输入键名")
        key_layout.addWidget(self.key_input)

        value_layout = QVBoxLayout()
        value_layout.addWidget(QLabel("值:"))
        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("输入值或JSON")
        value_layout.addWidget(self.value_input)

        type_layout = QVBoxLayout()
        type_layout.addWidget(QLabel("类型:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["字符串", "整数", "浮点数", "布尔值", "JSON"])
        type_layout.addWidget(self.type_combo)

        input_layout.addLayout(key_layout)
        input_layout.addLayout(value_layout)
        input_layout.addLayout(type_layout)
        operation_layout.addLayout(input_layout)

        # 添加按钮
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("添加/更新")
        self.add_button.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.add_button.clicked.connect(self.add_or_update_item)

        self.delete_button = QPushButton("删除")
        self.delete_button.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        self.delete_button.clicked.connect(self.delete_item)

        self.clear_button = QPushButton("清空内存")
        self.clear_button.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.clear_button.clicked.connect(self.clear_memory)

        self.copy_button = QPushButton("复制值")
        self.copy_button.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.copy_button.clicked.connect(self.copy_selected_value)

        self.reload_button = QPushButton("重新加载")
        self.reload_button.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")
        self.reload_button.clicked.connect(self.reload_data)

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.reload_button)
        operation_layout.addLayout(button_layout)

        operation_group.setLayout(operation_layout)
        main_layout.addWidget(operation_group)

        # 添加格式提示
        format_tips = QLabel("格式提示: 布尔值输入 true/false; JSON输入 {'key': 'value'}")
        format_tips.setStyleSheet("color: #7f8c8d; font-style: italic;")
        main_layout.addWidget(format_tips)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QTableWidget {
                background-color: white;
                alternate-background-color: #f8f9fa;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 4px;
                border: 1px solid #2980b9;
            }
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #bdc3c7;
            }
            QPushButton:hover {
                background-color: #d6dbdf;
            }
            QLineEdit {
                padding: 5px;
            }
            QLabel[accessibleName="status"] {
                color: #27ae60;
                font-weight: bold;
            }
        """)

        self.status_label.setAccessibleName("status")

        # 设置交替行颜色
        self.table.setAlternatingRowColors(True)

        # 添加状态栏
        self.statusBar().showMessage("就绪")

    def handle_selection_changed(self):
        """处理表格选择变化"""
        selected_items = self.table.selectedItems()
        if selected_items:
            # 获取选中的键
            row = selected_items[0].row()
            self.current_selected_key = self.table.item(row, 0).text()
        else:
            self.current_selected_key = None

    def reload_data(self):
        """重新加载数据"""
        self.statusBar().showMessage("重新加载数据...")
        self.load_data()
        QTimer.singleShot(1000, lambda: self.statusBar().showMessage("数据已重新加载"))

    def show_context_menu(self, pos):
        """显示右键上下文菜单"""
        menu = QMenu()

        # 创建菜单项
        copy_key_action = QAction("复制键", self)
        copy_value_action = QAction("复制值", self)
        copy_row_action = QAction("复制整行", self)
        edit_value_action = QAction("编辑值", self)

        # 连接到处理函数
        copy_key_action.triggered.connect(self.copy_selected_key)
        copy_value_action.triggered.connect(self.copy_selected_value)
        copy_row_action.triggered.connect(self.copy_selected_row)
        edit_value_action.triggered.connect(self.edit_selected_value)

        # 添加到菜单
        menu.addAction(copy_key_action)
        menu.addAction(copy_value_action)
        menu.addAction(copy_row_action)
        menu.addAction(edit_value_action)

        # 显示菜单
        menu.exec_(self.table.viewport().mapToGlobal(pos))

    def load_data(self):
        """从共享内存加载数据并更新UI"""
        try:
            # 保存当前滚动条位置
            v_scroll_value = self.table.verticalScrollBar().value()
            h_scroll_value = self.table.horizontalScrollBar().value()

            # 获取内存信息
            file_size = os.path.getsize(self.mem_file) if os.path.exists(self.mem_file) else 0
            used = self.shared_dict._get_used_size()
            capacity = self.shared_dict._get_data_capacity()
            items_count = len(self.shared_dict)

            # 更新信息标签
            self.size_label.setText(f"大小: {file_size / 1024:.1f} KB (已用: {used} B, 容量: {capacity} B)")
            self.items_label.setText(f"项目: {items_count}")
            self.status_label.setText("状态: 已连接")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")

            # 如果没有数据，清空表格并返回
            if items_count == 0:
                self.table.setRowCount(0)
                self.statusBar().showMessage("共享内存为空")
                self.current_selected_key = None
                return

            # 获取所有键值对
            items = list(self.shared_dict.items())

            # 设置表格行数
            self.table.setRowCount(items_count)

            # 填充表格
            for i, (key, value) in enumerate(items):
                # 键列
                key_item = QTableWidgetItem(key)
                key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
                key_item.setToolTip("双击复制键")
                self.table.setItem(i, 0, key_item)

                # 值列
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False)
                    value_type = "JSON"
                else:
                    value_str = str(value)
                    value_type = type(value).__name__

                value_item = QTableWidgetItem(value_str)
                value_item.setToolTip("双击复制值")
                self.table.setItem(i, 1, value_item)

                # 类型列
                type_item = QTableWidgetItem(value_type)
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, 2, type_item)

                # 根据值类型设置颜色
                if isinstance(value, bool):
                    value_item.setBackground(QColor(230, 255, 230))
                elif isinstance(value, (int, float)):
                    value_item.setBackground(QColor(230, 230, 255))
                elif isinstance(value, dict):
                    value_item.setBackground(QColor(255, 230, 230))
                elif isinstance(value, list):
                    value_item.setBackground(QColor(255, 255, 200))

            # 恢复选中状态
            if self.current_selected_key:
                for row in range(self.table.rowCount()):
                    if self.table.item(row, 0).text() == self.current_selected_key:
                        self.table.selectRow(row)
                        break

            # 恢复滚动条位置
            self.table.verticalScrollBar().setValue(v_scroll_value)
            self.table.horizontalScrollBar().setValue(h_scroll_value)

            self.statusBar().showMessage(f"已加载 {items_count} 个项目")

        except Exception as e:
            self.statusBar().showMessage(f"加载数据时出错: {str(e)}")
            self.status_label.setText("状态: 错误")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def add_or_update_item(self):
        """添加或更新键值对"""
        key = self.key_input.text().strip()
        value_str = self.value_input.text().strip()
        value_type = self.type_combo.currentText()

        if not key:
            QMessageBox.warning(self, "输入错误", "键不能为空")
            return

        if not value_str:
            QMessageBox.warning(self, "输入错误", "值不能为空")
            return

        try:
            # 根据选择的类型转换值
            if value_type == "字符串":
                value = value_str
            elif value_type == "整数":
                value = int(value_str)
            elif value_type == "浮点数":
                value = float(value_str)
            elif value_type == "布尔值":
                value = value_str.lower() in ["true", "1", "yes", "y"]
            elif value_type == "JSON":
                value = json.loads(value_str)

            # 更新共享内存
            self.shared_dict[key] = value
            self.statusBar().showMessage(f"已更新键: {key}")

            # 更新选中状态为新添加/更新的键
            self.current_selected_key = key

            # 清空输入框
            self.key_input.clear()
            self.value_input.clear()

            # 刷新数据
            self.load_data()

        except ValueError:
            QMessageBox.warning(self, "输入错误", f"无法将 '{value_str}' 转换为 {value_type}")
        except json.JSONDecodeError:
            QMessageBox.warning(self, "JSON错误", "无效的JSON格式")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新共享内存时出错: {str(e)}")

    def delete_item(self):
        """删除选中的键值对"""
        if not self.current_selected_key:
            QMessageBox.warning(self, "选择错误", "请先选择一个键值对")
            return

        key = self.current_selected_key

        # 确认删除
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要删除键 '{key}' 吗?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                if key in self.shared_dict:
                    del self.shared_dict[key]
                    self.statusBar().showMessage(f"已删除键: {key}")
                    self.current_selected_key = None  # 清除选中状态
                    self.load_data()
                else:
                    QMessageBox.warning(self, "删除错误", f"键 '{key}' 不存在")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除键值时出错: {str(e)}")

    def clear_memory(self):
        """清空共享内存"""
        reply = QMessageBox.question(self, "确认清空",
                                     "确定要清空共享内存中的所有数据吗?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                self.shared_dict.clear()
                self.statusBar().showMessage("已清空共享内存")
                self.current_selected_key = None  # 清除选中状态
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"清空共享内存时出错: {str(e)}")

    def copy_selected_key(self):
        """复制选中的键"""
        if self.current_selected_key:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.current_selected_key)
            self.statusBar().showMessage(f"已复制键: {self.current_selected_key}")
        else:
            QMessageBox.warning(self, "操作错误", "请先选择一个键值对")

    def copy_selected_value(self):
        """复制选中的值"""
        if self.current_selected_key:
            try:
                value = self.shared_dict[self.current_selected_key]
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = str(value)

                clipboard = QApplication.clipboard()
                clipboard.setText(value_str)
                self.statusBar().showMessage(f"已复制值: {value_str[:50]}{'...' if len(value_str) > 50 else ''}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"复制值时出错: {str(e)}")
        else:
            QMessageBox.warning(self, "操作错误", "请先选择一个键值对")

    def copy_selected_row(self):
        """复制整行数据"""
        if self.current_selected_key:
            try:
                value = self.shared_dict[self.current_selected_key]
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = str(value)

                clipboard = QApplication.clipboard()
                clipboard.setText(f"{self.current_selected_key} = {value_str}")
                self.statusBar().showMessage(
                    f"已复制整行: {self.current_selected_key} = {value_str[:50]}{'...' if len(value_str) > 50 else ''}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"复制行数据时出错: {str(e)}")
        else:
            QMessageBox.warning(self, "操作错误", "请先选择一个键值对")

    def edit_selected_value(self):
        """编辑选中的值"""
        if not self.current_selected_key:
            return

        key = self.current_selected_key

        try:
            current_value = self.shared_dict.get(key)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取值时出错: {str(e)}")
            return

        if current_value is None:
            QMessageBox.warning(self, "编辑错误", f"键 '{key}' 不存在")
            return

        # 创建编辑对话框
        dialog = QDialog(self)
        dialog.setWindowTitle(f"编辑值: {key}")
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(300)
        layout = QVBoxLayout()

        # 添加标签
        layout.addWidget(QLabel(f"键: {key}"))
        layout.addWidget(QLabel("新值:"))

        # 根据值的类型创建不同的输入控件
        if isinstance(current_value, (dict, list)):
            # 对于JSON使用多行文本编辑
            text_edit = QTextEdit()
            text_edit.setPlainText(json.dumps(current_value, indent=2, ensure_ascii=False))
            layout.addWidget(text_edit)
            input_widget = text_edit
        else:
            # 对于其他类型使用单行输入
            line_edit = QLineEdit()
            line_edit.setText(str(current_value))
            layout.addWidget(line_edit)
            input_widget = line_edit

        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            try:
                new_value = input_widget.text() if isinstance(input_widget, QLineEdit) else input_widget.toPlainText()

                # 尝试解析JSON
                if isinstance(current_value, (dict, list)):
                    self.shared_dict[key] = json.loads(new_value)
                else:
                    # 保持原始类型
                    if isinstance(current_value, bool):
                        self.shared_dict[key] = new_value.lower() in ["true", "1", "yes", "y"]
                    elif isinstance(current_value, int):
                        self.shared_dict[key] = int(new_value)
                    elif isinstance(current_value, float):
                        self.shared_dict[key] = float(new_value)
                    else:
                        self.shared_dict[key] = new_value

                self.statusBar().showMessage(f"已更新键: {key}")
                self.load_data()
            except Exception as e:
                QMessageBox.warning(self, "编辑错误", f"更新值失败: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 停止定时器
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        # 关闭共享内存映射
        if hasattr(self.shared_dict, '_map') and self.shared_dict._map:
            try:
                self.shared_dict._map.close()
            except:
                pass

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")

    # 设置调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(50, 50, 50))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = SharedMemoryViewer()
    window.show()
    sys.exit(app.exec_())