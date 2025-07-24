import sys
import random
import json
import os
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import requests
import qtawesome as pta
from scipy.io import loadmat, savemat
import numpy as np
import colorsys

# 导入米家API库
from mijiaAPI import mijiaLogin, mijiaAPI, mijiaDevice, get_device_info
from mijiaAPI.login import logger
from qrcode import QRCode
import matplotlib.pyplot as plt

from mijiaAPI.apis import mijiaAPI
from mijiaAPI.devices import mijiaDevice, get_device_info
from urllib import parse
from mijiaAPI.consts import defaultUA, msgURL, loginURL, qrURL, accountURL, defaultUA

import argparse

'''
Mijia interface for AssistBCI-v2025

Author: Li Haobo
Email: lihaoboece@gmail.com
'''


DEFAULT_DEVICE_CUSTOM_PATH = os.path.join(os.path.expanduser("~"), ".assistbci_config")
DEFAULT_MIJIA_USER_PATH = os.path.join(os.path.expanduser("~"), ".assistbci_config", "mijia-api-auth.json")
DEFAULT_SETTINGS_PATH = os.path.join(DEFAULT_DEVICE_CUSTOM_PATH, "device_mapping.json")
CLASSIFIER_DATA_PATH = os.path.join('../../demos', '..', 'assistbci_models', 'classifier')


class _mijiaLogin(mijiaLogin):
    def __init__(self, save_path):
        super().__init__(save_path)

    @staticmethod
    def _print_qr(loginurl: str, box_size: int = 10) -> None:
        """
        打印并保存二维码。

        Args:
            loginurl (str): 包含登录信息的URL。
            box_size (int, optional): 二维码大小。默认为10。
        """
        logger.info('请使用米家APP扫描下方二维码')
        qr = QRCode(border=1, box_size=box_size)
        qr.add_data(loginurl)
        img = qr.make_image()
        img_array = np.array(img)
        plt.ion()
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')  # 不显示坐标轴
        plt.pause(0.1)
        try:
            qr.print_ascii(invert=True, tty=True)
        except OSError:
            qr.print_ascii(invert=True, tty=False)
            logger.info('如果无法扫描二维码，'
                        '请更改终端字体，'
                        '如"Maple Mono"、"Fira Code"等。\n'
                        '或者直接使用当前目录下的qr.png文件。')


class LoginDialog(QDialog): #米家账号登陆
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("米家账号登录")
        self.setFixedSize(400, 300)
        self.auth_path = DEFAULT_MIJIA_USER_PATH

        layout = QVBoxLayout(self)

        # 登录方式选择
        method_group = QGroupBox("登录方式")
        method_layout = QVBoxLayout()
        self.qr_radio = QRadioButton("二维码登录")
        self.qr_radio.setChecked(True)
        self.account_radio = QRadioButton("账号密码登录")
        method_layout.addWidget(self.qr_radio)
        method_layout.addWidget(self.account_radio)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # 账号密码输入
        self.username_label = QLabel("账号:")
        self.username_input = QLineEdit()
        self.password_label = QLabel("密码:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        # 默认隐藏账号密码输入
        self.username_label.setVisible(False)
        self.username_input.setVisible(False)
        self.password_label.setVisible(False)
        self.password_input.setVisible(False)

        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        # 状态标签
        self.status_label = QLabel("准备登录...")
        layout.addWidget(self.status_label)

        # 按钮
        btn_layout = QHBoxLayout()
        login_btn = QPushButton("登录")
        login_btn.clicked.connect(self.do_login)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(login_btn)
        layout.addLayout(btn_layout)

        # 连接信号
        self.qr_radio.toggled.connect(self.toggle_login_method)

    def toggle_login_method(self, checked):
        if checked:
            # 二维码登录
            self.username_label.setVisible(False)
            self.username_input.setVisible(False)
            self.password_label.setVisible(False)
            self.password_input.setVisible(False)
        else:
            # 账号密码登录
            self.username_label.setVisible(True)
            self.username_input.setVisible(True)
            self.password_label.setVisible(True)
            self.password_input.setVisible(True)

    def do_login(self):
        try:
            if self.qr_radio.isChecked():
                self.status_label.setText("正在生成二维码...")
                QApplication.processEvents()

                # 二维码登录
                login_obj = _mijiaLogin(save_path=self.auth_path)
                auth_data = login_obj.QRlogin()

                self.status_label.setText("二维码登录成功!")
                plt.close()
                self.accept()
                return auth_data
            else:
                # 账号密码登录
                username = self.username_input.text()
                password = self.password_input.text()

                if not username or not password:
                    QMessageBox.warning(self, "输入错误", "请输入账号和密码")
                    return

                self.status_label.setText("正在登录...")
                QApplication.processEvents()

                # 账号密码登录
                login_obj = _mijiaLogin(save_path=self.auth_path)
                auth_data = login_obj.login(username, password)

                self.status_label.setText("登录成功!")
                plt.close()
                self.accept()
                return auth_data
        except Exception as e:
            QMessageBox.critical(self, "登录失败", f"登录过程中发生错误:\n{str(e)}")
            self.status_label.setText(f"登录失败: {str(e)}")
            return None


class BubbleWidget(QWidget):#状态圆形类
    def __init__(self, name, color, parent=None, diameter=60):
        super().__init__(parent)
        self.name = name
        self.color = color
        self.diameter = diameter
        self.target_diameter = self.diameter
        self.animation_direction = 1
        self.setFixedSize(120, 120)
        self.setAcceptDrops(True)
        self.devices = []  # 存储设备及其配置
        self.parent_window = parent  # 保存对主窗口的引用

        #删除
        self.delete_button = QPushButton(self)
        self.delete_button.setIcon(QApplication.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        self.delete_button.setFixedSize(24, 24)
        self.delete_button.setStyleSheet("""
                    QPushButton {
                        background: rgba(255, 255, 255, 150);
                        border-radius: 12px;
                        border: 1px solid #999;
                    }
                    QPushButton:hover {
                        background: rgba(255, 0, 0, 150);
                    }
                """)
        self.delete_button.hide()  # 初始隐藏
        self.delete_button.clicked.connect(self.show_delete_menu)

        # 连接鼠标事件
        self.setMouseTracking(True)

        # 添加执行按钮
        self.execute_button = QPushButton("执行", self)
        self.execute_button.setFixedSize(60, 30)
        self.execute_button.setStyleSheet("""
                   QPushButton {
                       background: rgba(0, 150, 0, 180);
                       color: white;
                       border-radius: 10px;
                       font-weight: bold;
                   }
                   QPushButton:hover {
                       background: rgba(0, 200, 0, 200);
                   }
               """)
        self.execute_button.hide()
        self.execute_button.clicked.connect(self.execute_actions)


        # 动画定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(100)

    def update_animation(self):
        # 呼吸动画效果
        if self.diameter >= self.target_diameter + 5:
            self.animation_direction = -1
        elif self.diameter <= self.target_diameter - 5:
            self.animation_direction = 1

        self.diameter += self.animation_direction * 0.5
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制小球
        center = self.rect().center()
        gradient = QRadialGradient(center, self.diameter / 2, center)
        gradient.setColorAt(0, QColor(255, 255, 255, 200))
        gradient.setColorAt(1, self.color)

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, self.diameter / 2, self.diameter / 2)

        # 绘制文字
        font = QFont("Microsoft YaHei", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(50, 50, 50))
        painter.drawText(self.rect(), Qt.AlignCenter, self.name)

        # 在右上角绘制设备数量标记
        if self.devices:
            painter.setBrush(QColor(255, 0, 0, 180))
            painter.setPen(Qt.NoPen)
            count_rect = QRect(self.width() - 30, 10, 20, 20)
            painter.drawEllipse(count_rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(count_rect, Qt.AlignCenter, str(len(self.devices)))


    def dragEnterEvent(self, event): #添加触发条件之后变大一点
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self.target_diameter = self.diameter + 1
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.target_diameter = self.diameter - 1

    def dropEvent(self, event):
        device_info = json.loads(event.mimeData().text())
        device_name = device_info['name']

        # 检查设备是否已添加
        if not any(device['name'] == device_name for device in self.devices):
            self.show_config_dialog(device_info)
        else:
            QMessageBox.warning(self, "设备已存在", f"设备 {device_name} 已经配置过此触发条件")

        self.target_diameter = self.diameter
        self.update()
        event.acceptProposedAction()

        # 通知主窗口更新状态
        if self.parent_window:
            self.parent_window.update_status()

    def show_config_dialog(self, device_info):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"配置 {device_info['name']} - {self.name}")
        dialog.setMinimumSize(600, 500)

        layout = QVBoxLayout(dialog)

        # 标题
        title = QLabel(f"当 <b>{self.name}</b> 发生时，执行以下操作：")
        title.setStyleSheet("font-size: 14px; margin-bottom: 15px;")
        layout.addWidget(title)

        # 设备信息
        device_label = QLabel(f"设备: {device_info['name']}  代号：{device_info['model']}")
        device_label.setStyleSheet("font-size: 12px; color: #555; margin-bottom: 10px;")
        layout.addWidget(device_label)

        # 功能选项
        options_group = QGroupBox("可执行操作")
        options_group.setStyleSheet("""
            QGroupBox {
                padding: 20px;  /* 内边距 */
                margin: 5px;    /* 外边距 */
            }
        """)
        options_layout = QVBoxLayout()

        # 创建一个滚动区域以容纳更多控件
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        try:
            device_data = get_device_info(device_info['model'])
            self.property_controls = {}  # 存储属性控件的字典
            self.checkboxes = []

            # 添加属性操作
            if device_data.get('properties'):
                prop_group = QGroupBox("属性控制")
                prop_layout = QVBoxLayout()

                for prop in device_data['properties']:
                    if 'w' in prop['rw']:  # 只显示可写属性
                        # 创建水平布局包含复选框和值设置控件
                        hbox = QHBoxLayout()

                        cb = QCheckBox(f"{prop['name']}")
                        cb.setProperty('type', 'property')
                        cb.setProperty('prop_name', prop['name'])
                        cb.setProperty('method', prop['method'])
                        cb.setToolTip(prop['description'])
                        self.checkboxes.append(cb)
                        hbox.addWidget(cb)

                        # 根据属性类型创建不同的值设置控件
                        value_widget = None
                        prop_type = prop.get('type', 'string')

                        if prop_type == 'bool':
                            # 布尔值使用下拉选择
                            combo = QComboBox()
                            combo.addItem("True", True)
                            combo.addItem("False", False)
                            combo.setEnabled(False)  # 初始禁用，直到复选框被选中
                            value_widget = combo

                        elif prop_type == 'enum' and 'values' in prop:
                            # 枚举值使用下拉选择
                            combo = QComboBox()
                            for value in prop['values']:
                                combo.addItem(str(value), value)
                            combo.setEnabled(False)
                            value_widget = combo

                        elif prop_type in ['int', 'float']:
                            # 数值类型使用滑动条和数字输入框
                            min_val = prop.get('min', 0)
                            max_val = prop.get('max', 100)

                            # 创建数字输入框
                            spin = QDoubleSpinBox() if prop_type == 'float' else QSpinBox()
                            spin.setRange(min_val, max_val)
                            spin.setValue(min_val)
                            spin.setEnabled(False)

                            # 创建滑动条
                            slider = QSlider(Qt.Horizontal)
                            slider.setRange(min_val, max_val)
                            slider.setValue(min_val)
                            slider.setEnabled(False)

                            # 连接滑动条和数字输入框
                            spin.valueChanged.connect(slider.setValue)
                            slider.valueChanged.connect(spin.setValue)

                            # 创建容器布局
                            value_layout = QVBoxLayout()
                            value_layout.addWidget(spin)
                            value_layout.addWidget(slider)

                            value_widget = QWidget()
                            value_widget.setLayout(value_layout)

                        else:
                            # 其他类型使用文本输入
                            line_edit = QLineEdit()
                            line_edit.setPlaceholderText("输入值...")
                            line_edit.setEnabled(False)
                            value_widget = line_edit

                        # 连接复选框状态改变事件
                        cb.stateChanged.connect(
                            lambda state, widget=value_widget:
                            widget.setEnabled(state == Qt.Checked)
                        )

                        # 保存控件引用
                        self.property_controls[prop['name']] = value_widget
                        hbox.addWidget(value_widget)
                        hbox.addStretch()

                        prop_layout.addLayout(hbox)

                prop_group.setLayout(prop_layout)
                scroll_layout.addWidget(prop_group)

            # 添加动作操作
            if device_data.get('actions'):
                action_group = QGroupBox("执行动作")
                action_layout = QVBoxLayout()

                for action in device_data['actions']:
                    cb = QCheckBox(f"{action['name']}")
                    cb.setProperty('type', 'action')
                    cb.setProperty('action_name', action['name'])
                    cb.setProperty('method', action['method'])
                    cb.setToolTip(action['description'])
                    action_layout.addWidget(cb)
                    self.checkboxes.append(cb)

                action_group.setLayout(action_layout)
                scroll_layout.addWidget(action_group)

            # 设置滚动区域内容
            scroll_area.setWidget(scroll_content)
            options_layout.addWidget(scroll_area)
            options_group.setLayout(options_layout)
            layout.addWidget(options_group)

            # 设备使用方法输入
            usage_label = QLabel("设备使用方法说明:")
            usage_label.setStyleSheet("font-size: 12px; margin-top: 10px;")
            layout.addWidget(usage_label)

            self.usage_edit = QTextEdit()
            self.usage_edit.setPlaceholderText("输入设备的使用方法说明...")
            self.usage_edit.setMaximumHeight(60)
            layout.addWidget(self.usage_edit)

            # 按钮
            btn_layout = QHBoxLayout()
            save_btn = QPushButton("保存配置")
            save_btn.setStyleSheet("background: #4CAF50; color: white; padding: 8px;")
            save_btn.clicked.connect(dialog.accept)

            cancel_btn = QPushButton("取消")
            cancel_btn.setStyleSheet("background: #f44336; color: white; padding: 8px;")
            cancel_btn.clicked.connect(dialog.reject)

            btn_layout.addStretch()
            btn_layout.addWidget(cancel_btn)
            btn_layout.addWidget(save_btn)
            layout.addLayout(btn_layout)

            if dialog.exec_() == QDialog.Accepted:
                selected_actions = []
                usage_text = self.usage_edit.toPlainText()

                # 收集选中的操作
                for cb in self.checkboxes:
                    if cb.isChecked():
                        action_type = cb.property('type')
                        action_info = {
                            'type': action_type,
                            'name': cb.text(),
                            'method': cb.property('method')
                        }

                        # 如果是属性操作，获取用户设置的值
                        if action_type == 'property':
                            prop_name = cb.property('prop_name')
                            value_widget = self.property_controls.get(prop_name)

                            # 根据控件类型获取值
                            if isinstance(value_widget, QComboBox):
                                action_info['value'] = value_widget.currentData()
                            elif isinstance(value_widget, QLineEdit):
                                action_info['value'] = value_widget.text()
                            elif isinstance(value_widget, QWidget):
                                # 数值类型 - 从数字输入框获取值
                                spin = value_widget.findChild(QSpinBox) or value_widget.findChild(QDoubleSpinBox)
                                if spin:
                                    action_info['value'] = spin.value()

                        selected_actions.append(action_info)

                if selected_actions:
                    QMessageBox.information(self, "配置成功", f"已设置当<b>{self.name}</b>时，{device_info['name']}执行: "
                                                              f"{len(selected_actions)}个操作")
                    # 保存设备配置信息
                    device_config = {
                        'name': device_info['name'],
                        'model': device_info['model'],
                        'did': device_info['did'],
                        'actions': selected_actions,
                        'usage': usage_text
                    }
                    self.devices.append(device_config)

                    # 通知主窗口更新状态
                    if self.parent_window:
                        self.parent_window.update_status()
                else:
                    QMessageBox.warning(self, "未选择操作", "请至少选择一个操作")

        except Exception as e:
            QMessageBox.critical(self, "设备信息获取失败", f"无法获取设备信息:\n{str(e)}")

    def resizeEvent(self, event):
        # 更新删除按钮位置到右上角
        # self.delete_button.move(self.width() - 30, 10)
        self.execute_button.move(self.width() - 70, self.height() - 40)
        super().resizeEvent(event)

    def enterEvent(self, event):
        # 鼠标进入时显示删除按钮
        self.delete_button.show()
        self.execute_button.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # 鼠标离开时隐藏删除按钮
        self.delete_button.hide()
        self.execute_button.hide()
        super().leaveEvent(event)

    def show_delete_menu(self):
        if not self.devices:
            return

        # 创建删除菜单
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: white;
                border: 1px solid #ccc;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px 5px 10px;
            }
            QMenu::item:selected {
                background: #e0f7fa;
            }
        """)

        # 添加删除所有选项
        delete_all_action = menu.addAction("删除所有配置")
        delete_all_action.triggered.connect(lambda: self.delete_device(-1))

        menu.addSeparator()

        # 添加每个设备的删除选项
        for i, device in enumerate(self.devices):
            action = menu.addAction(f"删除: {device['name']}")
            action.triggered.connect(lambda checked, idx=i: self.delete_device(idx))

        # 在按钮下方显示菜单
        menu.exec_(self.mapToGlobal(self.delete_button.pos() + QPoint(0, self.delete_button.height())))

    def delete_device(self, index):
        if index == -1:  # 删除所有
            self.devices.clear()
            QMessageBox.information(self, "删除成功", "已删除所有配置的设备")
        elif 0 <= index < len(self.devices):
            device_name = self.devices[index]['name']
            del self.devices[index]
            QMessageBox.information(self, "删除成功", f"已删除设备: {device_name}")

        self.update()

        # 通知主窗口更新状态
        if self.parent_window:
            self.parent_window.update_status()

    def execute_actions(self):
        """执行该状态的所有设备操作"""
        if not self.devices:
            QMessageBox.information(self, "无操作", "此状态未配置任何设备操作")
            return

        if self.parent_window:
            self.parent_window.execute_bubble_actions(self)


    def contextMenuEvent(self, event):
        """右键菜单事件"""
        menu = QMenu(self)

        # 添加删除状态选项
        delete_state_action = menu.addAction("删除此状态")
        delete_state_action.triggered.connect(self.delete_state)

        # 如果有设备配置，添加设备管理选项
        if self.devices:
            device_menu = menu.addMenu("管理设备")

            # 添加删除所有设备选项
            delete_all_action = device_menu.addAction("删除所有设备")
            delete_all_action.triggered.connect(lambda: self.delete_device(-1))

            # 添加删除单个设备选项
            for i, device in enumerate(self.devices):
                action = device_menu.addAction(f"删除: {device['name']}")
                action.triggered.connect(lambda checked, idx=i: self.delete_device(idx))

        # 显示菜单
        menu.exec_(event.globalPos())

    def delete_state(self):
        """删除当前状态"""
        if self.parent_window:
            self.parent_window.delete_bubble_state(self)


class DeviceListWidget(QListWidget):
    device_icons = {
        "light": "fa5.lightbulb",  # 灯泡（Solid）
        "ac": "fa5.snowflake",  # 雪花（Solid）
        "curtain": "fa5.columns",  # 列（Solid）
        "lock": "fa5s.lock",  # 锁（Solid）
        "tv": "fa5s.tv",  # 电视（Solid）
        "speaker": "fa5.speaker",  # 扬声器（Solid）
        "air": "fa5.wind",  # 风（Solid）
        "camera": "fa5.camera",  # 相机（Solid）
        "plug": "fa5s.plug",  # 插头（Solid）
        "alarm": "fa5.bell",  # 铃铛（Solid）
        "default": "fa5s.desktop"  # 电脑（Solid，更常用）
    }

    device_keywords = [
        (("light", "灯"), "light"),
        (("ac", "空调"), "ac"),
        (("curtain", "窗帘"), "curtain"),
        (("lock", "门锁"), "lock"),
        (("tv", "电视"), "tv"),
        (("speaker", "音箱"), "speaker"),
        (("air", "净化器"), "air"),
        (("camera", "摄像头"), "camera"),
        (("plug", "插座"), "plug"),
        (("alarm", "闹钟", "clock"), "alarm")
    ]
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setViewMode(QListView.IconMode)
        self.setIconSize(QSize(60, 60))
        self.setSpacing(15)
        self.setAcceptDrops(False)
        self.setDropIndicatorShown(False)
        self.setStyleSheet("""
            QListWidget {
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
            }
            QListWidget::item {
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                color: black;  /* 默认文字颜色 */
            }
            QListWidget::item:hover {
                background: #e0f7fa;
                border: 1px solid #4dd0e1;
            }
            QListWidget::item:selected {
                background: #e0f7fa;
                border: 1px solid #4dd0e1;
                color: black;
                font-weight: bold;
            }
        """)

    def add_device(self, device_info):
        name = device_info['name']
        item = QListWidgetItem(name)

        # 遍历关键字匹配设备类型
        icon = None
        for keywords, device_type in self.device_keywords:
            # 检查设备型号或名称中是否包含任一关键字（不区分大小写）
            if any(
                    keyword in device_info.get('model', '').lower()
                    or keyword in name.lower()
                    for keyword in keywords
            ):
                # 根据设备类型获取对应图标，未找到则使用默认图标
                icon = pta.icon(self.device_icons.get(device_type, self.device_icons["default"]))
                break

        # 未匹配到任何设备类型时返回默认图标
        if icon is None:
            icon = pta.icon(self.device_icons["default"])
        item.setIcon(icon)
        item.setData(Qt.UserRole, json.dumps(device_info))
        self.addItem(item)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item:
            device_info = json.loads(item.data(Qt.UserRole))
            mime_data = QMimeData()
            mime_data.setText(json.dumps(device_info))

            drag = QDrag(self)
            drag.setMimeData(mime_data)

            # 创建拖动时的设备缩略图
            pixmap = QPixmap(80, 80)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(200, 230, 255, 200))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, 80, 80, 15, 15)
            painter.drawPixmap(10, 10, item.icon().pixmap(60, 60))
            painter.setPen(QColor(50, 50, 50))
            painter.drawText(QRect(0, 65, 80, 15), Qt.AlignCenter, item.text())
            painter.end()

            drag.setPixmap(pixmap)
            drag.setHotSpot(QPoint(40, 40))
            drag.exec_(Qt.MoveAction)


class Mijia_home(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能家居自动化配置")
        self.setGeometry(100, 100, 1000, 600)
        self.current_settings_file = None  # 当前保存的设置文件路径
        self.api = None  # 米家API对象
        self.auth_path = DEFAULT_MIJIA_USER_PATH

        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 左侧区域 - 气泡
        left_frame = QFrame()
        left_frame.setStyleSheet("""
            QFrame {
                background: #e1e4ea;
                border-radius: 12px;
            }
        """) # border: 2px dashed #81d4fa;
        left_layout = QVBoxLayout(left_frame)
        left_layout.setAlignment(Qt.AlignTop)

        title = QLabel("触发状态")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #5d7db3;
                padding: 10px;
            }
        """) #border-bottom: 2px solid #81d4fa;
        left_layout.addWidget(title)

        # 气泡容器
        bubble_container = QWidget()
        bubble_container.setMinimumSize(400, 500)
        self.bubble_layout = QGridLayout(bubble_container)
        self.bubble_layout.setAlignment(Qt.AlignCenter)
        self.bubble_layout.setSpacing(30)

        # 创建气泡
        self.bubbles = []  # 保存所有气泡的引用
        self.create_bubbles()

        scroll_area = QScrollArea()
        scroll_area.setWidget(bubble_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        left_layout.addWidget(scroll_area)

        # 右侧区域 - 设备列表
        right_frame = QFrame()
        right_frame.setStyleSheet("""
            QFrame {
                background: #f9f9f9;
                border-radius: 12px;
                border: 2px solid #e0e0e0;
            }
        """)
        right_layout = QVBoxLayout(right_frame)

        # 顶部工具栏
        toolbar_layout = QHBoxLayout()

        refresh_btn = QPushButton("刷新设备")
        refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.clicked.connect(self.load_mijia_devices)
        refresh_btn.setToolTip("从米家账号重新加载设备列表")

        login_btn = QPushButton("重新登录")
        login_btn.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        login_btn.clicked.connect(self.show_login_dialog)
        login_btn.setToolTip("重新登录米家账号")

        toolbar_layout.addWidget(refresh_btn)
        toolbar_layout.addWidget(login_btn)
        toolbar_layout.addStretch()

        right_layout.addLayout(toolbar_layout)

        title = QLabel("智能设备")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #5d7db3;
                padding: 10px;
                border-bottom: 2px solid #e0e0e0;
            }
        """)
        right_layout.addWidget(title)

        # 设备列表
        self.device_list = DeviceListWidget()
        right_layout.addWidget(self.device_list)

        # 状态统计信息
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 12px; color: #666;")

        # 旧映射导入
        if os.path.exists(DEFAULT_SETTINGS_PATH):
            self._load_settings(DEFAULT_SETTINGS_PATH, auto=True)

        # 初始化状态
        self.update_status()
        right_layout.addWidget(self.status_label)

        # 添加执行进度对话框
        self.progress_dialog = None

        # 添加到主布局
        main_layout.addWidget(left_frame, 1)
        main_layout.addWidget(right_frame, 1)

        # 创建菜单栏
        self.create_menu_bar()

        # 状态栏
        self.statusBar().showMessage("拖放设备到触发条件上进行配置")

        # 尝试自动登录
        self.try_auto_login()


    def try_auto_login(self):
        if os.path.exists(self.auth_path):
            try:
                with open(self.auth_path, 'r') as f:
                    auth_data = json.load(f)

                self.api = mijiaAPI(auth_data=auth_data)
                if self.api.available:
                    self.load_mijia_devices()
                    self.statusBar().showMessage("自动登录成功，已加载设备列表")
                    return True
            except Exception as e:
                self.statusBar().showMessage(f"自动登录失败: {str(e)}")
        return False

    def show_login_dialog(self):
        login_dialog = LoginDialog(self)
        if login_dialog.exec_() == QDialog.Accepted:
            try:
                # 重新加载设备列表
                self.load_mijia_devices()
                self.statusBar().showMessage("登录成功，已加载设备列表")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载设备列表时出错:\n{str(e)}")

    def load_mijia_devices(self):
        if not self.api:
            if not self.try_auto_login():
                self.show_login_dialog()
                return

        try:
            # 清空现有设备列表
            self.device_list.clear()

            # 获取设备列表
            devices = self.api.get_devices_list()

            # 添加到设备列表
            for device in devices:
                if device.get('isOnline', False):
                    device_info = {
                        'name': device['name'],
                        'model': device['model'],
                        'did': device['did'],
                        'online': True
                    }
                    self.device_list.add_device(device_info)

            self.statusBar().showMessage(f"已加载 {len(devices)} 个设备")
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载设备列表时出错:\n{str(e)}")

    def create_menu_bar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # save_action = QAction('保存设置', self) ##不开放，系统自动保存
        # save_action.triggered.connect(self.save_settings)
        # file_menu.addAction(save_action)

        save_as_action = QAction('另存为...', self)
        save_as_action.triggered.connect(self.save_settings_as)
        file_menu.addAction(save_as_action)

        load_action = QAction('加载设置', self)
        load_action.triggered.connect(self.load_settings)
        file_menu.addAction(load_action)

        export_action = QAction('导出配置', self)
        export_action.triggered.connect(self.export_settings)
        file_menu.addAction(export_action)

        # 米家菜单
        mijia_menu = menubar.addMenu('米家')

        login_action = QAction('登录账号', self)
        login_action.triggered.connect(self.show_login_dialog)
        mijia_menu.addAction(login_action)

        refresh_action = QAction('刷新设备', self)
        refresh_action.triggered.connect(self.load_mijia_devices)
        mijia_menu.addAction(refresh_action)

    def auto_save(self):
        if not os.path.exists(DEFAULT_DEVICE_CUSTOM_PATH):  # 如果路径不存在
            os.makedirs(DEFAULT_DEVICE_CUSTOM_PATH)
        self._save_to_file(DEFAULT_SETTINGS_PATH)

    def save_settings_as(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存设置", "", "JSON Files (*.json);;All Files (*)", options=options)

        if file_name:
            if not file_name.endswith('.json'):
                file_name += '.json'
            self.current_settings_file = file_name
            self._save_to_file(file_name)
            QMessageBox.information(self, "保存成功", f"设置已保存到 {file_name}")
        self.update_status()


    def _load_settings(self, file_name, auto=False):
        try:
            with open(file_name, 'r') as f:
                settings = json.load(f)

            # 清除所有气泡的当前设备
            for bubble in self.bubbles:
                bubble.devices.clear()
                bubble.update()

            # 加载配置
            for bubble_name, devices in settings.items():
                for bubble in self.bubbles:
                    if bubble.name == bubble_name:
                        bubble.devices = devices
                        bubble.update()
                        break
            if not auto:
                self.current_settings_file = file_name
            self.update_status()
            return 0
        except Exception as e:
            return e

    def load_settings(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "加载设置", "", "JSON Files (*.json);;All Files (*)", options=options)

        if file_name:
            state = self._load_settings(file_name)

            if state:
                QMessageBox.warning(self, "加载失败", f"无法加载设置: {str(state)}")
            else:
                QMessageBox.information(self, "加载成功", f"设置已从 {file_name} 加载")

    def export_settings(self):
        # 获取所有配置
        settings = self._get_current_settings()

        # 创建导出文本
        export_text = "AssistBCI自动化配置\n===================\n\n"
        export_text += f"导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for bubble_name, devices in settings.items():
            export_text += f"触发条件: {bubble_name}\n"
            export_text += f"已配置设备: {len(devices)}个\n"

            for device in devices:
                export_text += f"\n- 设备: {device['name']} ({device['model']})\n"
                export_text += f"  执行操作:\n"

                for action in device['actions']:
                    if action['type'] == 'property':
                        export_text += f"    • 设置属性: {action['name']}\n"
                    else:
                        export_text += f"    • 执行动作: {action['name']}\n"

                if device['usage']:
                    export_text += f"  使用方法: {device['usage']}\n"

            export_text += "\n" + "-" * 30 + "\n"

        # 显示导出对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("导出配置")
        dialog.setFixedSize(600, 500)

        layout = QVBoxLayout(dialog)

        # 添加导出文本编辑框
        text_edit = QTextEdit()
        text_edit.setPlainText(export_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        # 添加按钮
        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("复制到剪贴板")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(export_text))

        save_btn = QPushButton("保存为文本文件")
        save_btn.clicked.connect(lambda: self._save_export_text(export_text))

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)

        btn_layout.addWidget(copy_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec_()

    def _save_export_text(self, text):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存导出文本", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "保存成功", f"导出文本已保存到 {file_name}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"无法保存文件: {str(e)}")

    def _save_to_file(self, file_name):
        settings = self._get_current_settings()

        try:
            with open(file_name, 'w') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存设置: {str(e)}")
            return False

    def _get_current_settings(self):
        settings = {}
        for bubble in self.bubbles:
            if bubble.devices:
                settings[bubble.name] = bubble.devices
        return settings


    @staticmethod
    def get_state():
        classifier_labels = []
        if os.path.exists(CLASSIFIER_DATA_PATH):
            for filename in os.listdir(CLASSIFIER_DATA_PATH):
                if filename.endswith('.mat'):
                    _name = os.path.join(CLASSIFIER_DATA_PATH, filename)
                    _data = loadmat(_name)
                    classifier_labels.append(_data['labels'])
            if len(classifier_labels) == 0:
                return None, None
            return np.unique(np.concatenate(classifier_labels, axis=0), return_counts=True)

    @staticmethod
    def del_state(label):
        if os.path.exists(CLASSIFIER_DATA_PATH):
            for filename in os.listdir(CLASSIFIER_DATA_PATH):
                if filename.endswith('.mat'):
                    _name = os.path.join(CLASSIFIER_DATA_PATH, filename)
                    _data = loadmat(_name)
                    keep_index = _data['labels'] != label
                    classifier_data = _data['data'][keep_index,...]
                    classifier_labels = _data['labels'][keep_index,...]
                    if classifier_labels.shape[0] == 0:
                        print("删除状态后，文件：", _name, "为空，将被删除")
                        os.remove(_name)
                    else:
                        savemat(_name, {'data': classifier_data, 'labels': classifier_labels})
        else:
            return -1


    @staticmethod
    def generate_colors(n):
        colors = []
        for i in range(n):
            hue = i / n  # 均匀分布色相（0-1）
            saturation = 0.7  # 高饱和度
            lightness = 0.6  # 中等明度
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def create_bubbles(self):
        BUBBLES_NUM_COLUM = 3
        MAX_BUBBLE_DIAMETER = 100

        states, counts = self.get_state()
        if states is None:
            states = np.array([])
            counts = np.array([])

        diameters = (counts / np.sum(counts)) * MAX_BUBBLE_DIAMETER #根据标记数量计算状态球的大小
        colors = self.generate_colors(len(states))
        conditions = [(state, QColor(colors[i])) for i, state in enumerate(states)]

        # 将气泡添加到布局中
        positions = [(i//BUBBLES_NUM_COLUM, i%BUBBLES_NUM_COLUM) for i in range(len(conditions))]

        for (row, col), (name, color), diameter in zip(positions, conditions, diameters):
            bubble = BubbleWidget(name, color, self, int(diameter))  # 传递self作为parent_window
            self.bubble_layout.addWidget(bubble, row, col, Qt.AlignCenter)
            self.bubbles.append(bubble)  # 保存气泡引用

    def update_status(self):
        """更新状态栏统计信息"""
        total_devices = 0
        device_names = set()

        for bubble in self.bubbles:
            total_devices += len(bubble.devices)
            for device in bubble.devices:
                device_names.add(device['name'])

        status_text = f"当前配置状态:\n"
        status_text += f"• 已配置触发条件: {len([b for b in self.bubbles if b.devices])}/{len(self.bubbles)}\n"
        status_text += f"• 已配置设备总数: {total_devices}\n"
        status_text += f"• 涉及不同设备: {len(device_names)}种\n"

        # 添加当前设置文件信息
        if self.current_settings_file:
            status_text += f"\n当前设置文件: {self.current_settings_file}"
        else:
            status_text += f"\n当前设置文件: 自动保存"

        self.status_label.setText(status_text)
        self.auto_save()


    def execute_bubble_actions(self, bubble):
        """执行气泡中的所有设备操作"""
        if not self.api:
            if not self.try_auto_login():
                QMessageBox.warning(self, "未登录", "请先登录米家账号")
                return

        # 创建进度对话框
        self.progress_dialog = QProgressDialog(
            f"正在执行 '{bubble.name}' 状态的操作...",
            "取消",
            0,
            len(bubble.devices),
            self
        )
        self.progress_dialog.setWindowTitle("执行操作")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)

        # 执行所有设备操作
        for i, device_config in enumerate(bubble.devices):
            if self.progress_dialog.wasCanceled():
                break

            self.progress_dialog.setLabelText(f"正在操作: {device_config['name']}")
            self.progress_dialog.setValue(i)
            QApplication.processEvents()

            try:
                # 创建设备对象
                device = mijiaDevice(
                    api=self.api,
                    dev_name=device_config['name'],
                    did=device_config['did']
                )

                # 执行所有配置的操作
                for action in device_config['actions']:
                    if self.progress_dialog.wasCanceled():
                        break

                    if action['type'] == 'action':
                        # 执行动作
                        device.run_action(
                            name=action['name'],
                            method=action['method']
                        )
                        self.statusBar().showMessage(f"执行 {device_config['name']} 的 {action['name']} 成功")

                    elif action['type'] == 'property':
                        # 设置属性
                        device.set(
                            name=action['name'],
                            value=action['value'],
                            # did=action['method']
                        )
                        self.statusBar().showMessage(
                            f"设置 {device_config['name']} 的 {action['name']} 为 {action['value']} 成功")

                    # 添加短暂延迟
                    time.sleep(0.5)

            except Exception as e:
                error_msg = f"执行设备 {device_config['name']} 操作时出错:\n{str(e)}"
                QMessageBox.warning(self, "执行错误", error_msg)
                self.statusBar().showMessage(f"执行 {device_config['name']} 操作失败: {str(e)}")

        # 完成进度
        self.progress_dialog.setValue(len(bubble.devices))
        QMessageBox.information(self, "执行完成", f"'{bubble.name}' 状态的所有操作已执行完成！")


    def delete_bubble_state(self, bubble):
        """删除指定状态气泡"""
        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要永久删除状态 '{bubble.name}' 吗？\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        try:
            # 从模型文件中删除状态
            self.del_state(bubble.name)

            # 从界面中移除气泡
            self.bubble_layout.removeWidget(bubble)
            bubble.deleteLater()
            self.bubbles.remove(bubble)

            # 重新创建气泡布局
            self.recreate_bubbles()

            QMessageBox.information(self, "删除成功", f"状态 '{bubble.name}' 已永久删除")
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "删除失败", f"删除状态时出错:\n{str(e)}")

    def recreate_bubbles(self):
        """重新创建所有气泡"""
        # 保存当前配置
        current_settings = self._get_current_settings()

        # 清除现有气泡
        for bubble in self.bubbles:
            self.bubble_layout.removeWidget(bubble)
            bubble.deleteLater()
        self.bubbles.clear()

        # 创建新气泡
        states, counts = self.get_state()
        if states is None:
            states = np.array([])
            counts = np.array([])

        colors = self.generate_colors(len(states))
        conditions = [(state, QColor(colors[i])) for i, state in enumerate(states)]

        # 将气泡添加到布局中
        BUBBLES_NUM_COLUM = 3
        MAX_BUBBLE_DIAMETER = 100
        diameters = (counts / np.sum(counts)) * MAX_BUBBLE_DIAMETER if counts.size > 0 else []
        positions = [(i // BUBBLES_NUM_COLUM, i % BUBBLES_NUM_COLUM) for i in range(len(conditions))]

        for (row, col), (name, color), diameter in zip(positions, conditions, diameters):
            bubble = BubbleWidget(name, color, self, int(diameter))
            self.bubble_layout.addWidget(bubble, row, col, Qt.AlignCenter)
            self.bubbles.append(bubble)

            # 恢复配置
            if name in current_settings:
                bubble.devices = current_settings[name]
                bubble.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 设置全局字体（例如：微软雅黑，12号）
    font = QFont("Microsoft YaHei", 12)  # 字体名称，字号
    app.setFont(font)  # 应用到整个应用程序

    window = Mijia_home()
    window.show()
    sys.exit(app.exec_())

