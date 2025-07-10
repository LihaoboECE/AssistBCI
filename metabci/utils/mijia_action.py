import json
import os
import time
from mijiaAPI import mijiaAPI, mijiaDevice

# 全局配置文件路径
from metabci.utils.mijia_connect_home import DEFAULT_MIJIA_USER_PATH, DEFAULT_SETTINGS_PATH


'''
Mijia API for AssistBCI-v2025

also can be simply used for every worker 
in metabci by demo.brainstim_demos.sharedmemory

Author: Li Haobo
Email: lihaoboece@gmail.com
'''


def load_settings(settings_path=DEFAULT_SETTINGS_PATH):
    """加载自动化配置设置"""
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"配置文件不存在: {settings_path}")

    with open(settings_path, 'r') as f:
        return json.load(f)


def get_state_actions(settings, state_label):
    """获取特定状态的所有操作配置"""
    for state, devices in settings.items():
        if state == state_label:
            return devices
    return None


def execute_state_actions(state_label, settings_path=DEFAULT_SETTINGS_PATH, auth_path=DEFAULT_MIJIA_USER_PATH):
    """
    执行特定状态的所有自动化操作

    :param state_label: 状态标签（如"焦虑"、"放松"等）
    :param settings_path: 自动化配置文件的路径
    :param auth_path: 米家认证信息的路径
    """
    try:
        # 1. 加载配置
        settings = load_settings(settings_path)

        # 2. 获取特定状态的操作
        state_actions = get_state_actions(settings, state_label)
        if not state_actions:
            print(f"未找到状态 '{state_label}' 的配置")
            return False

        # 3. 加载米家认证信息
        if not os.path.exists(auth_path):
            raise FileNotFoundError(f"米家认证文件不存在: {auth_path}")

        with open(auth_path, 'r') as f:
            auth_data = json.load(f)

        # 4. 初始化米家API
        api = mijiaAPI(auth_data=auth_data)
        if not api.available:
            print("米家API初始化失败")
            return False

        # 5. 执行所有操作
        print(f"开始执行 '{state_label}' 状态的所有操作")
        print(f"共找到 {len(state_actions)} 个设备配置")

        for i, device_config in enumerate(state_actions):
            device_name = device_config['name']
            print(f"\n[{i + 1}/{len(state_actions)}] 操作设备: {device_name}")

            # 创建设备对象
            device = mijiaDevice(
                api=api,
                dev_name=device_name,
                did=device_config['did']
            )

            # 执行所有配置的操作
            for action in device_config['actions']:
                try:
                    if action['type'] == 'action':
                        # 执行动作
                        print(f"  → 执行动作: {action['name']}")
                        device.run_action(
                            name=action['name'],
                            method=action['method']
                        )
                    elif action['type'] == 'property':
                        # 设置属性
                        print(f"  → 设置属性: {action['name']} = {action.get('value', 'N/A')}")
                        device.set(
                            name=action['name'],
                            value=action['value'],
                        )

                    # 添加短暂延迟
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  执行失败: {str(e)}")

        print(f"\n'{state_label}' 状态的所有操作执行完成")
        return True

    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        return False


# 示例使用方式
if __name__ == "__main__":
    # 执行"中性"状态的所有操作
    execute_state_actions("中性")
