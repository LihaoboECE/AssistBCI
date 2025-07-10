import sys
import os
import time
from PyQt5.QtGui import (QIcon, QPixmap, QFont, QColor, QPainter,
                         QLinearGradient, QBrush)
from PyQt5.QtCore import (Qt, QTimer, QPropertyAnimation, QRect,
                          QPoint, QEasingCurve, pyqtSignal, QEvent)
from metabci.utils.mijia_connect_home import Mijia_home
from metabci.utils.sharedmemory import SharedDict
from metabci.utils.sharedmemory_ManageTool import SharedMemoryViewer
from PyQt5.QtWidgets import (QApplication, QPushButton, QVBoxLayout,
                             QWidget, QLabel, QHBoxLayout,
                             QFrame, QLineEdit, QComboBox,
                             QGraphicsDropShadowEffect, QListWidgetItem,
                             QListWidget, QSystemTrayIcon, QMenu,
                             QDialog, QCheckBox, QGroupBox, QButtonGroup,
                             QRadioButton)
from metabci.utils.mijia_action import execute_state_actions
import json
from datetime import datetime, timedelta

'''
Mean interface for AssistBCI-v2025

Author: Li Haobo
Email: lihaoboece@gmail.com
'''

CLASSIFIER_DATA_PATH = os.path.join('..', '..', 'assistbci_models', 'classifier')
NOTIFICATION_HISTORY_PATH = os.path.join('..', '..', 'assistbci_models', 'notification_history.json')

# ======================= 通知 =======================
class NotificationManager:
    """单例通知管理器，确保一次只显示一个通知并保存历史"""
    _instance = None
    current_notification = None
    history = []  # 存储历史通知
    MAX_HISTORY = 50  # 最大历史记录数量

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NotificationManager, cls).__new__(cls)
            # 尝试从文件加载历史记录
            try:
                with open(NOTIFICATION_HISTORY_PATH, 'r') as f:
                    cls.history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                cls.history = []
        return cls._instance

    def show_notification(self, notification):
        """显示通知，如果已有通知则先关闭"""
        if self.current_notification:
            self.current_notification.close_notification()

        self.current_notification = notification
        notification.show_notification()

    def notification_closed(self, notification):
        """通知关闭时的回调"""
        self.current_notification = None
        # 保存到历史记录
        self.add_to_history(notification)

    def add_to_history(self, notification):
        """添加通知到历史记录"""
        # 创建通知记录
        record = {
            'timestamp': datetime.now().isoformat(),
            'type': notification.__class__.__name__,
            'content': notification.get_content()
        }

        # 添加到历史记录
        self.history.insert(0, record)  # 最新的在最前面

        # 限制历史记录大小
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[:self.MAX_HISTORY]

        # 保存到文件
        self.save_history()

    def save_history(self):
        """保存历史记录到文件"""
        try:
            with open(NOTIFICATION_HISTORY_PATH, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"保存通知历史失败: {e}")

    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self.save_history()


class ElegantNotification(QWidget):
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(350)

        # 添加阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.shadow.setOffset(0, 3)
        self.setGraphicsEffect(self.shadow)

        # 动画设置
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(350)
        self.animation.setEasingCurve(QEasingCurve.OutBack)

        # 自动关闭定时器
        self.close_timer = QTimer()
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.close_notification)

        # 用户交互标志
        self.user_interacting = False
        self.operation_completed = False

        # 通知管理器
        self.notification_manager = NotificationManager()

    def paintEvent(self, event):
        """绘制圆角背景和渐变颜色"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 渐变背景
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255))
        gradient.setColorAt(1, QColor(245, 245, 245))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)

        # 绘制圆角矩形
        rect = self.rect()
        rect.adjust(1, 1, -1, -1)  # 留出阴影空间
        painter.drawRoundedRect(rect, 12, 12)

    def show_notification(self, timeout=3000):
        """显示通知"""
        # 获取主屏幕
        primary_screen = QApplication.primaryScreen()
        if not primary_screen:
            return

        screen_geometry = primary_screen.availableGeometry()
        start_pos = QPoint(screen_geometry.right(), screen_geometry.bottom() - self.height() - 20)
        end_pos = QPoint(screen_geometry.right() - self.width() - 20, screen_geometry.bottom() - self.height() - 20)

        self.move(start_pos)
        self.show()

        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.start()

        # 重置状态
        self.user_interacting = False
        self.operation_completed = False

        # 启动自动关闭定时器
        if timeout > 0:
            self.close_timer.start(timeout)

    def close_notification(self):
        """关闭通知"""
        # 如果用户正在交互且操作未完成，不关闭
        if self.user_interacting and not self.operation_completed:
            return

        self.animation.finished.connect(self.close)
        primary_screen = QApplication.primaryScreen()
        if not primary_screen:
            return

        screen_geometry = primary_screen.availableGeometry()
        end_pos = QPoint(screen_geometry.right(), self.y())

        self.animation.setStartValue(self.pos())
        self.animation.setEndValue(end_pos)
        self.animation.start()
        self.closed.emit()


    def get_content(self):
        """获取通知内容，子类应重写此方法"""
        return {"message": "未实现的内容"}

    def enterEvent(self, event):
        """鼠标进入通知区域"""
        self.user_interacting = True
        self.close_timer.stop()  # 停止自动关闭
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开通知区域"""
        self.user_interacting = False
        super().leaveEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:
            if not self.isActiveWindow():
                self.close_notification()
        super().changeEvent(event)



    def complete_operation(self):
        """标记操作已完成"""
        self.operation_completed = True
        self.notification_manager.notification_closed(self)
        # 如果用户不在交互中，立即关闭
        if not self.user_interacting:
            self.close_notification()


class StatusNotification(ElegantNotification):
    status_selected = pyqtSignal(str)
    status_rejected = pyqtSignal()

    def __init__(self, existing_statuses, real_time):
        super().__init__()
        self.existing_statuses = existing_statuses
        self.real_time = real_time
        self.setup_ui()


    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 15)
        main_layout.setSpacing(15)

        # 标题区域
        title_layout = QHBoxLayout()

        # 标题
        title_label = QLabel("状态更新")
        title_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 消息内容
        message_label = QLabel(f"{int(time.time() - self.real_time)}秒前 检测到新的状态，请选择或输入状态名称")
        message_label.setStyleSheet("color: #34495e; font-size: 13px; line-height: 1.4;")
        message_label.setWordWrap(True)

        # 输入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入新状态名称...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #dfe6e9;
                border-radius: 6px;
                font-size: 13px;
                color: #2d3436;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        self.input_field.installEventFilter(self)  # 监听输入框事件

        # 现有状态列表
        status_list_label = QLabel("选择现有状态:")
        status_list_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")

        self.status_combo = QComboBox()
        self.status_combo.addItems(self.existing_statuses)
        self.status_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #dfe6e9;
                border-radius: 6px;
                font-size: 13px;
                color: #2d3436;
                min-height: 36px;
            }
            QComboBox::drop-down {
                width: 30px;
                border: none;
            }
        """)
        self.status_combo.installEventFilter(self)  # 监听下拉框事件

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(10)

        # 否按钮
        reject_btn = QPushButton("取消")
        reject_btn.setCursor(Qt.PointingHandCursor)
        reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #992d22;
            }
        """)
        reject_btn.clicked.connect(self.on_reject)

        # 确认按钮
        confirm_btn = QPushButton("确认")
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        confirm_btn.clicked.connect(self.on_confirm)

        button_layout.addWidget(reject_btn)
        button_layout.addStretch()
        button_layout.addWidget(confirm_btn)

        # 添加到主布局
        main_layout.addLayout(title_layout)
        main_layout.addWidget(message_label)
        main_layout.addWidget(self.input_field)
        main_layout.addWidget(status_list_label)
        main_layout.addWidget(self.status_combo)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def eventFilter(self, obj, event):
        """事件过滤器，用于检测用户交互"""
        if event.type() in [QEvent.Enter, QEvent.Leave, QEvent.MouseButtonPress, QEvent.KeyPress]:
            self.user_interacting = True
            self.close_timer.stop()  # 停止自动关闭
        return super().eventFilter(obj, event)

    def on_reject(self):
        self.status_rejected.emit()
        self.complete_operation()  # 标记操作完成
        self.close_notification()

    def on_confirm(self):
        if self.input_field.text():
            self.status_selected.emit(self.input_field.text())
        else:
            self.status_selected.emit(self.status_combo.currentText())
        self.complete_operation()  # 标记操作完成
        self.close_notification()


    def get_content(self):
        """获取通知内容"""
        return {
            "title": "状态更新",
            "message": "检测到新的状态，请选择或输入状态名称",
            "input_text": self.input_field.text(),
            "selected_status": self.status_combo.currentText()
        }


class ReportNotification(ElegantNotification):
    report_action = pyqtSignal(str)

    def __init__(self, report_message):
        super().__init__()
        self.report_message = report_message
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 15)
        main_layout.setSpacing(15)

        # 标题区域
        title_layout = QHBoxLayout()

        # 标题
        title_label = QLabel("状态报告")
        title_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 消息内容
        message_label = QLabel(self.report_message)
        message_label.setStyleSheet("color: #34495e; font-size: 13px; line-height: 1.4;")
        message_label.setWordWrap(True)

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(10)

        # 查看详情按钮
        action_btn = QPushButton("执行自动化")
        action_btn.setCursor(Qt.PointingHandCursor)
        action_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2471a3;
            }
        """)
        action_btn.clicked.connect(lambda: self.on_action("action"))

        # 忽略按钮
        false_btn = QPushButton("错误")
        false_btn.setCursor(Qt.PointingHandCursor)
        false_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7a89;
            }
        """)
        false_btn.clicked.connect(lambda: self.on_action("pred_false"))

        button_layout.addStretch()
        button_layout.addWidget(action_btn)
        button_layout.addWidget(false_btn)

        # 添加到主布局
        main_layout.addLayout(title_layout)
        main_layout.addWidget(message_label)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def on_action(self, action):
        self.report_action.emit(action)
        self.complete_operation()  # 标记操作完成
        self.close_notification()


    def get_content(self):
        """获取通知内容"""
        return {
            "title": "状态报告",
            "message": self.report_message
        }


class HistoryNotificationDialog(QDialog):
    def __init__(self, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("历史通知")
        self.setMinimumSize(600, 500)

        # 保存完整历史记录
        self.full_history = history
        self.filtered_history = history.copy()

        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 标题和筛选按钮
        title_layout = QHBoxLayout()
        title_label = QLabel("历史通知")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        title_layout.addWidget(title_label)

        # 添加筛选按钮
        filter_btn = QPushButton("筛选")
        filter_btn.setFixedSize(60, 30)
        filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        filter_btn.clicked.connect(self.show_filter_dialog)
        title_layout.addWidget(filter_btn)

        layout.addLayout(title_layout)

        # 筛选信息显示
        self.filter_info_label = QLabel("显示所有通知")
        self.filter_info_label.setFont(QFont("Microsoft YaHei", 9))
        self.filter_info_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.filter_info_label)

        # 列表控件
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setStyleSheet("""
            QListWidget::item:alternate {
                background-color: #f0f4f8;
            }
        """)

        # 填充历史记录
        self.populate_history(self.filtered_history)

        # 按钮区域
        button_layout = QHBoxLayout()

        # 清空按钮
        clear_btn = QPushButton("清空历史")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        clear_btn.clicked.connect(self.clear_history)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        close_btn.clicked.connect(self.accept)

        button_layout.addStretch()
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(close_btn)

        # 添加到主布局
        layout.addWidget(self.list_widget)
        layout.addLayout(button_layout)

    def show_filter_dialog(self):
        """显示筛选对话框"""
        filter_dialog = QDialog(self)
        filter_dialog.setWindowTitle("筛选通知")
        filter_dialog.setFixedSize(300, 400)
        filter_dialog.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 8px;
            }
            QLabel {
                color: #333;
            }
        """)

        layout = QVBoxLayout(filter_dialog)
        # layout.setContentsMargins(20, 20, 20, 20)

        # 通知类型筛选
        type_group = QGroupBox("通知类型")
        type_layout = QVBoxLayout()

        self.type_all = QRadioButton("所有通知")
        self.type_all.setChecked(True)
        self.type_status = QRadioButton("状态更新")
        self.type_report = QRadioButton("状态报告")

        type_layout.addWidget(self.type_all)
        type_layout.addWidget(self.type_status)
        type_layout.addWidget(self.type_report)
        type_group.setLayout(type_layout)

        # 时间范围筛选
        time_group = QGroupBox("时间范围")
        time_layout = QVBoxLayout()

        self.time_all = QRadioButton("所有时间")
        self.time_all.setChecked(True)
        self.time_today = QRadioButton("今天")
        self.time_week = QRadioButton("最近7天")

        time_layout.addWidget(self.time_all)
        time_layout.addWidget(self.time_today)
        time_layout.addWidget(self.time_week)
        time_group.setLayout(time_layout)

        # 按钮区域
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("应用")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
            }
        """)
        apply_btn.clicked.connect(lambda: self.apply_filters(filter_dialog))

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
            }
        """)
        cancel_btn.clicked.connect(filter_dialog.reject)

        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(type_group)
        layout.addWidget(time_group)
        layout.addLayout(btn_layout)

        filter_dialog.exec_()

    def apply_filters(self, dialog):
        """应用筛选条件"""
        # 确定通知类型
        if self.type_status.isChecked():
            filter_type = "StatusNotification"
        elif self.type_report.isChecked():
            filter_type = "ReportNotification"
        else:
            filter_type = None

        # 确定时间范围
        if self.time_today.isChecked():
            today = datetime.now().date()
            time_filter = lambda ts: datetime.fromisoformat(ts).date() == today
        elif self.time_week.isChecked():
            week_ago = datetime.now().date() - timedelta(days=7)
            time_filter = lambda ts: datetime.fromisoformat(ts).date() >= week_ago
        else:
            time_filter = lambda ts: True

        # 应用筛选
        self.filtered_history = [
            record for record in self.full_history
            if (filter_type is None or record['type'] == filter_type) and time_filter(record['timestamp'])
        ]

        # 更新显示
        self.populate_history(self.filtered_history)

        # 更新筛选信息
        type_text = "所有通知" if filter_type is None else (
            "状态更新" if filter_type == "StatusNotification" else "状态报告")

        if self.time_today.isChecked():
            time_text = "今天"
        elif self.time_week.isChecked():
            time_text = "最近7天"
        else:
            time_text = "所有时间"

        self.filter_info_label.setText(f"显示: {type_text} | 时间: {time_text} | 共 {len(self.filtered_history)} 条")

        dialog.accept()

    def populate_history(self, history_to_show):
        """填充历史记录到列表"""
        self.list_widget.clear()

        if not history_to_show:
            item = QListWidgetItem("没有历史通知")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.list_widget.addItem(item)
            return

        for record in history_to_show:
            # 解析时间
            try:
                dt = datetime.fromisoformat(record['timestamp'])
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = "未知时间"

            # 创建列表项
            item = QListWidgetItem()
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(10, 5, 10, 5)
            layout.setSpacing(5)

            # 标题和时间
            header_layout = QHBoxLayout()

            title_label = QLabel(record['content'].get('title', '通知'))
            title_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            title_label.setStyleSheet("color: #2c3e50;")

            time_label = QLabel(time_str)
            time_label.setFont(QFont("Microsoft YaHei", 8))
            time_label.setStyleSheet("color: #7f8c8d;")

            header_layout.addWidget(title_label)
            header_layout.addStretch()
            header_layout.addWidget(time_label)

            # 内容
            content_label = QLabel(record['content'].get('message', '无内容'))
            content_label.setFont(QFont("Microsoft YaHei", 9))
            content_label.setStyleSheet("color: #34495e;")
            content_label.setWordWrap(True)

            # 对于状态通知，显示额外信息
            if record['type'] == 'StatusNotification':
                input_text = record['content'].get('input_text', '')
                selected_status = record['content'].get('selected_status', '')

                if input_text:
                    status_info = f"输入状态: {input_text}"
                elif selected_status:
                    status_info = f"选择状态: {selected_status}"
                else:
                    status_info = "未选择状态"

                status_label = QLabel(status_info)
                status_label.setFont(QFont("Microsoft YaHei", 8))
                status_label.setStyleSheet("color: #3498db; font-style: italic;")
                layout.addWidget(status_label)

            layout.addLayout(header_layout)
            layout.addWidget(content_label)

            # 设置列表项
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def clear_history(self):
        """清空历史记录"""
        NotificationManager().clear_history()
        self.full_history = []
        self.filtered_history = []
        self.populate_history([])
        self.filter_info_label.setText("显示所有通知 | 共 0 条")


# ======================= 主面板 =======================

class ClickableCard(QFrame):
    """可点击的功能卡片"""
    clicked = pyqtSignal()

    def __init__(self, title, description, color, parent=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.color = color
        self.setMinimumHeight(80)
        self.setCursor(Qt.PointingHandCursor)

        # 设置初始样式
        self.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 10px;
                border-left: 2px solid {color};
                padding: 5px;
            }}
        """)
        self.original_style = self.styleSheet()

        # 创建布局和内容
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {color};")

        desc_label = QLabel(description)
        desc_label.setFont(QFont("Microsoft YaHei", 9))
        desc_label.setStyleSheet("color: #666666;")
        desc_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)

    def mousePressEvent(self, event):
        """鼠标按下时的效果"""
        if event.button() == Qt.LeftButton:
            # 添加点击效果
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    border-left: 4px solid {self.color};
                    padding: 5px;
                }}
            """)
            self.clicked.emit()

    def mouseReleaseEvent(self, event):
        """鼠标释放时恢复原样式"""
        self.setStyleSheet(self.original_style)


class SettingsDialog(QDialog):
    """设置对话框，允许用户配置通知选项"""

    def __init__(self, parent=None, notifications_enabled=True, notification_type='all'):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setFixedSize(400, 800)
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 15px;
                padding: 10px;
                font-weight: bold;
                color: #333;
            }
            QLabel {
                color: #333;
            }
        """)

        self.notifications_enabled = notifications_enabled  # 默认开启通知
        self.notification_type = notification_type  # 默认接收所有通知

        # 保存父窗口引用
        self._buffer = SharedDict()

        # 新增设备控制设置
        self.device_control_enabled = True
        self.selected_device = ""
        self.selected_worker = ""

        # 初始化UI
        self.init_ui()

        # 添加刷新定时器
        self.status_monitor = QTimer(self)
        self.status_monitor.timeout.connect(self.check_update)
        self.status_monitor.start(1000)  # 每秒检查一次

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QLabel("系统设置")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0; margin: 10px 0;")
        layout.addWidget(separator)

        # 通知设置组
        notification_group = QGroupBox("通知设置")
        notification_layout = QVBoxLayout()

        # 启用通知复选框
        self.notify_checkbox = QCheckBox("启用通知")
        self.notify_checkbox.setChecked(self.notifications_enabled)
        self.notify_checkbox.setStyleSheet("QCheckBox { color: #333; }")
        notification_layout.addWidget(self.notify_checkbox)

        # 通知类型设置
        type_layout = QVBoxLayout()
        type_layout.setContentsMargins(20, 10, 0, 10)

        type_label = QLabel("通知类型:")
        type_label.setStyleSheet("color: #666;")
        type_layout.addWidget(type_label)

        # 通知类型单选按钮组
        self.type_button_group = QButtonGroup(self)

        # 所有通知
        all_radio = QRadioButton("接收所有通知")
        all_radio.setChecked(self.notification_type == "all")
        self.type_button_group.addButton(all_radio, 1)
        type_layout.addWidget(all_radio)

        # 仅状态更新
        status_radio = QRadioButton("仅接收状态更新")
        status_radio.setChecked(self.notification_type == "status")
        self.type_button_group.addButton(status_radio, 2)
        type_layout.addWidget(status_radio)

        # 仅报告通知
        report_radio = QRadioButton("仅接收报告通知")
        report_radio.setChecked(self.notification_type == "report")
        self.type_button_group.addButton(report_radio, 3)
        type_layout.addWidget(report_radio)

        notification_layout.addLayout(type_layout)
        notification_group.setLayout(notification_layout)
        layout.addWidget(notification_group)

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(100, 30)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        # 确定按钮
        ok_btn = QPushButton("确定")
        ok_btn.setFixedSize(100, 30)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # ================ 新增设备控制部分 ================
        device_group = QGroupBox("设备控制")
        device_layout = QVBoxLayout()

        # 设备选择下拉框
        device_label = QLabel("选择设备:")
        device_layout.addWidget(device_label)

        self.device_combo = QComboBox()
        # 从共享内存获取设备列表
        device_list = self._buffer.get('device_list', [])
        self.device_combo.addItems(device_list)
        device_layout.addWidget(self.device_combo)

        # 连接/断开按钮
        self.device_btn = QPushButton("连接设备")
        self.device_btn.setStyleSheet(self.get_button_style())
        self.device_btn.clicked.connect(self.toggle_device_connection)
        device_layout.addWidget(self.device_btn)

        # 设备状态显示
        self.device_status = QLabel("状态: 未连接")
        device_layout.addWidget(self.device_status)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # ================ 新增算法控制部分 ================
        algo_group = QGroupBox("算法控制")
        algo_layout = QVBoxLayout()

        # 算法开关
        self.algo_checkbox = QCheckBox("启用算法")
        self.algo_checkbox.setChecked(self._buffer.get('AG_control', 'disable') == 'enable')
        self.algo_checkbox.stateChanged.connect(self.toggle_algorithm)
        algo_layout.addWidget(self.algo_checkbox)

        # 算法选择下拉框
        worker_label = QLabel("选择算法:")
        algo_layout.addWidget(worker_label)

        self.worker_combo = QComboBox()
        # 从共享内存获取算法列表
        worker_list = self._buffer.get('worker_list', [])
        self.worker_combo.addItems(worker_list)
        algo_layout.addWidget(self.worker_combo)

        # 注册/注销按钮
        self.worker_btn = QPushButton("注册算法")
        self.worker_btn.setStyleSheet(self.get_button_style())
        self.worker_btn.clicked.connect(self.toggle_worker_registration)
        algo_layout.addWidget(self.worker_btn)

        # 算法状态显示
        self.worker_status = QLabel("状态: 未注册")
        algo_layout.addWidget(self.worker_status)

        # 启动/停止按钮
        self.worker_action_btn = QPushButton("启动算法")
        self.worker_action_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.worker_action_btn.clicked.connect(self.toggle_worker_action)
        self.worker_action_btn.setEnabled(False)
        algo_layout.addWidget(self.worker_action_btn)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)


    def get_button_style(self, color="#5e72e4"):
        """获取按钮样式"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #{QColor(color).darker(120).name()};
            }}
            QPushButton:disabled {{
                background-color: #95a5a6;
            }}
        """
    def check_update(self):
        # 更新设备状态
        self.update_device_status()
        # 更新算法状态
        self.update_worker_status()

    def update_device_status(self):
        """更新设备状态显示"""
        device_state = self._buffer.get('device_state', 'not_connected')
        if device_state == 'connected':
            self.device_status.setText("状态: 已连接")
            self.device_btn.setText("断开设备")
        else:
            self.device_status.setText("状态: 未连接")
            self.device_btn.setText("连接设备")

    def update_worker_status(self):
        """更新算法状态显示"""
        current_workers = self._buffer.get('current_workers', [])
        if current_workers is None:
            current_workers = []
        worker_name = self.worker_combo.currentText()

        if worker_name in current_workers:
            self.worker_status.setText(f"状态: {worker_name}已注册")
            self.worker_btn.setText("注销算法")
            self.worker_action_btn.setEnabled(True)

            # 检查算法是否正在运行
            if self._buffer.get('start_worker', False):
                self.worker_action_btn.setText("停止算法")
            else:
                self.worker_action_btn.setText("启动算法")
        else:
            self.worker_status.setText("状态: 未注册")
            self.worker_btn.setText("注册算法")
            self.worker_action_btn.setEnabled(False)
            self.worker_action_btn.setText("启动算法")

    def toggle_device_connection(self):
        """切换设备连接状态"""
        device_name = self.device_combo.currentText()
        current_device = self._buffer.get('connect_device', None)

        if current_device is None or current_device != device_name:
            # 连接设备
            self._buffer['connect_device'] = device_name
        else:
            # 断开设备
            self._buffer['connect_device'] = None

        # 更新状态
        # self.update_device_status()

    def toggle_worker_registration(self):
        """切换算法注册状态"""
        worker_name = self.worker_combo.currentText()
        current_workers = self._buffer.get('current_workers', None)
        if current_workers is None:
            current_workers = []

        if worker_name in current_workers:
            # 注销算法
            self._buffer['unreg_worker'] = worker_name
        else:
            # 注册算法
            self._buffer['reg_worker'] = worker_name

        # 更新状态
        # QTimer.singleShot(500, self.update_worker_status)  # 稍等片刻再更新状态

    def toggle_worker_action(self):
        """启动/停止算法"""
        if self._buffer.get('start_worker', False):
            # 停止算法
            self._buffer['stop_worker'] = True
        else:
            # 启动算法
            self._buffer['start_worker'] = True

        # 更新状态
        # QTimer.singleShot(500, self.update_worker_status)  # 稍等片刻再更新状态

    def toggle_algorithm(self, state):
        """切换算法开关状态"""
        if state == Qt.Checked:
            self._buffer['AG_control'] = 'enable'
        else:
            self._buffer['AG_control'] = 'disable'

    def get_settings(self):
        """获取当前设置"""
        self.notifications_enabled = self.notify_checkbox.isChecked()

        # 获取选择的通知类型
        checked_id = self.type_button_group.checkedId()
        if checked_id == 1:
            self.notification_type = "all"
        elif checked_id == 2:
            self.notification_type = "status"
        elif checked_id == 3:
            self.notification_type = "report"

        return self.notifications_enabled, self.notification_type


class SidePanelApp(QWidget):
    def __init__(self):
        super().__init__()

        # 重要设置，面板与后端信息传输共享内存
        self._buffer = SharedDict()
        self._buffer['AG_feedback'] = {}
        self.last_ag_timestamp = 0  # 记录上次的时间戳
        self.notifications_enabled = True  # 默认开启通知
        self.notification_type = "all"  # 默认接收所有通知

        self.processing_notification = None

        # 初始化UI
        self.init_ui()

        # 创建系统托盘图标
        self.init_tray_icon()

        # 初始位置在屏幕右侧外部
        screen_geo = QApplication.primaryScreen().geometry()
        self.setGeometry(
            screen_geo.width(),  # 初始X位置在屏幕右侧外部
            screen_geo.height() // 4,  # 垂直居中
            350,  # 宽度
            min(600, screen_geo.height() * 0.7)  # 高度
        )

        # 动画控制变量
        self.is_panel_visible = False
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)  # 动画时长300ms
        self.animation.setEasingCurve(QEasingCurve.OutCubic)  # 平滑的动画曲线

        # 添加状态监控定时器
        self.status_monitor = QTimer(self)
        self.status_monitor.timeout.connect(self.check_ag_status)
        self.status_monitor.start(2000)  # 每秒检查一次

    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle("Side Panel")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 创建主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建左侧内容区域
        content_widget = QWidget()
        content_widget.setObjectName("contentWidget")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)

        # 设置内容区域样式
        content_widget.setStyleSheet("""
            #contentWidget {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                border: 1px solid #e0e0e0;
            }
        """)

        # 创建标题
        title = QLabel("AssistBCI v2025")
        title_font = QFont("Microsoft YaHei", 16, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #333333;")

        # 创建副标题
        subtitle = QLabel("您的个人状态管理助手\nPowered by MetaBCI")
        subtitle.setFont(QFont("Microsoft YaHei", 10))
        subtitle.setStyleSheet("color: #333333;")

        # 创建副副标题
        subsubtitle = QLabel("University of Macau\nAuthor: Li Haobo\nEmail: lihaoboece@gmail.com")
        subsubtitle.setFont(QFont("Microsoft YaHei", 9))
        subsubtitle.setStyleSheet("color: #999999;")

        # 创建分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0; margin: 5px 0;")

        # 创建功能卡片
        card_layout = QVBoxLayout()
        card_layout.setSpacing(15)

        # 添加几个功能卡片
        cards = [
            ("设置", "设备连接/算法设置等", "#11cdef", self.open_settings),
            ("通知", "查看最近通知", "#2dce89", self.show_notifications),
            ("状态与自动化", "管理状态，智能家居自动化配置，自定义状态触发操作", "#9c27b0", self.open_smart_home_config),
            ("开发者工具", "查看/更改系统寄存器", "#fb6340", self.open_tools)
        ]

        for text, description, color, handler in cards:
            card = self.create_card(text, description, color)
            card.clicked.connect(handler)  # 连接点击事件处理函数
            card_layout.addWidget(card)

        # 添加关闭按钮
        close_btn = QPushButton("关闭面板")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #5e72e4;
                color: white;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a5bbf;
            }
        """)
        close_btn.clicked.connect(self.toggle_panel)

        # 添加所有组件到内容布局
        content_layout.addWidget(title)
        content_layout.addWidget(subtitle)
        content_layout.addWidget(subsubtitle)
        content_layout.addWidget(separator)
        content_layout.addWidget(close_btn)
        content_layout.addLayout(card_layout)

        # 将内容区和控制区添加到主布局
        main_layout.addWidget(content_widget)

        # 设置窗口大小
        self.setFixedSize(390, 600)

    def create_card(self, title, description, color):
        """创建功能卡片"""
        return ClickableCard(title, description, color)

    def init_tray_icon(self):
        """初始化系统托盘图标"""
        # 创建托盘图标
        self.tray_icon = QSystemTrayIcon(self)

        # 创建托盘图标菜单
        tray_menu = QMenu()

        # 添加菜单项
        show_action = tray_menu.addAction("显示面板")
        show_action.triggered.connect(self.toggle_panel)

        # 通知设置菜单项
        notify_action = tray_menu.addAction("通知设置")
        notify_action.triggered.connect(self.open_settings)

        tray_menu.addSeparator()

        exit_action = tray_menu.addAction("退出")
        exit_action.triggered.connect(self.close_app)

        # 设置托盘图标和菜单
        self.tray_icon.setIcon(QIcon(self.create_tray_icon()))
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)

        # 显示托盘图标
        self.tray_icon.show()

    def create_tray_icon(self):
        """动态创建托盘图标"""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制圆形背景
        painter.setBrush(QColor(94, 114, 228))  # #5e72e4
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)

        # 绘制应用图标
        painter.setPen(Qt.white)
        painter.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "BCI")

        painter.end()

        return pixmap

    def tray_icon_activated(self, reason):
        """托盘图标被激活时的处理"""
        if reason == QSystemTrayIcon.Trigger:
            self.toggle_panel()

    def toggle_panel(self):
        """切换面板显示状态"""
        # 获取当前屏幕（面板所在的屏幕）
        current_screen = QApplication.screenAt(self.geometry().center())
        if not current_screen:
            current_screen = QApplication.primaryScreen()

        screen_geo = current_screen.geometry()
        panel_width = self.width()

        if not self.is_panel_visible:
            # 显示面板
            target_geo = QRect(
                screen_geo.right() - panel_width,  # X位置
                screen_geo.height() // 4,  # Y位置
                panel_width,
                self.height()
            )
            self.animation.setStartValue(self.geometry())
            self.animation.setEndValue(target_geo)
            self.show()
            self.raise_()
            self.animation.start()
            self.is_panel_visible = True
        else:
            # 隐藏面板
            target_geo = QRect(
                screen_geo.right(),
                self.y(),
                panel_width,
                self.height()
            )
            self.animation.setStartValue(self.geometry())
            self.animation.setEndValue(target_geo)
            self.animation.start()
            self.is_panel_visible = False

    def close_app(self):
        """关闭应用程序"""
        self.tray_icon.hide()
        QApplication.quit()

    def mousePressEvent(self, event):
        """实现窗口拖动功能"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """实现窗口拖动功能"""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    # ===== 卡片点击事件处理函数 =====
    def open_settings(self):
        """打开设置对话框"""
        settings_dialog = SettingsDialog(self, self.notifications_enabled, self.notification_type)
        settings_dialog.exec_()

        # 获取用户设置
        self.notifications_enabled, self.notification_type = settings_dialog.get_settings()


    def get_notification_type_text(self):
        """获取通知类型的文本描述"""
        if self.notification_type == "all":
            return "所有通知"
        elif self.notification_type == "status":
            return "仅状态更新"
        elif self.notification_type == "report":
            return "仅报告通知"
        return "未知"

    def show_notifications(self):
        """显示历史通知对话框"""
        history_dialog = HistoryNotificationDialog(NotificationManager().history, self)
        history_dialog.exec_()

    def open_tools(self):
        """打开工具功能"""
        self.CMD_window = SharedMemoryViewer()
        self.CMD_window.show()

    def open_smart_home_config(self):
        """打开智能家居自动化配置页面"""
        self.smart_home_window = Mijia_home()
        self.smart_home_window.show()

    def check_ag_status(self):
        """检查AG_predict状态变化并显示通知"""
        try:
            # 如果通知被禁用，直接返回
            if not self.notifications_enabled:
                return

            # 确保AG_predict字段存在
            if 'AG_predict' not in self._buffer:
                return

            self.processing_notification = self._buffer['AG_predict']

            # 检查时间戳是否更新
            current_timestamp = int(self.processing_notification.get('timestamp', 0))
            if current_timestamp <= self.last_ag_timestamp:
                return

            # 更新时间戳
            self.last_ag_timestamp = current_timestamp

            # 根据设置的通知类型和new/report字段决定是否显示通知
            new_status = self.processing_notification.get('new', False)
            report_status = self.processing_notification.get('report', False)

            # 状态更新通知
            if new_status and self.notification_type in ["all", "status"]:
                self.show_status_notification()

            # 报告通知
            elif report_status and self.notification_type in ["all", "report"]:
                labels = self.processing_notification.get('labels', [])
                print(labels)
                report_text = f"{int(time.time() - int(self.processing_notification['timestamp']/1000 -5))}秒前 检测到状态: {','.join(labels)}"
                self.show_report_notification(report_text)

        except Exception as e:
            print(f"Error checking AG status: {e}")

    def show_status_notification(self):
        """显示状态更新通知"""
        # 示例状态列表

        statuses = self.processing_notification['labels']
        existing_statuses, _ = Mijia_home.get_state()
        for state in existing_statuses:
            if state not in statuses:
                statuses.append(state)

        notification = StatusNotification(statuses, int(self.processing_notification['timestamp']/1000 - 5))
        notification.status_selected.connect(self.handle_status_selection)
        notification.status_rejected.connect(lambda: print("用户取消了状态更新"))

        # 通过通知管理器显示通知
        notification.notification_manager.show_notification(notification)

    def show_report_notification(self, message):
        """显示报告通知"""
        notification = ReportNotification(message)
        notification.report_action.connect(self.handle_report_action)

        # 通过通知管理器显示通知
        notification.notification_manager.show_notification(notification)

    def handle_status_selection(self, status):
        """处理用户选择的状态"""
        print(f"用户选择了状态: {status}")
        print(self.processing_notification['timestamp'])

        _feedback = self._buffer['AG_feedback']
        _feedback[self.processing_notification['timestamp']] = status
        self._buffer['AG_feedback'] = _feedback

    def handle_report_action(self, action):
        """处理报告通知的操作"""
        print(action)
        if action == "action":
            execute_state_actions(self.processing_notification['labels'][0])
            _feedback = self._buffer['AG_feedback']
            _feedback[self.processing_notification['timestamp']] = True
            self._buffer['AG_feedback'] = _feedback

        elif action == "pred_false":
            _feedback = self._buffer['AG_feedback']
            _feedback[self.processing_notification['timestamp']] = False
            self._buffer['AG_feedback'] = _feedback


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 设置应用程序样式
    app.setStyle("Fusion")
    # 设置全局字体（例如：微软雅黑，12号）
    font = QFont("Microsoft YaHei", 12)  # 字体名称，字号
    app.setFont(font)  # 应用到整个应用程序

    # 创建并显示应用
    window = SidePanelApp()

    sys.exit(app.exec_())