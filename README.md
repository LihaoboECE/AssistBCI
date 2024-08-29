# AssistBCI——通用脑机接口辅助系统：基于 MetaBCI 的高效二次开发框架与全面个性化应用解决方案
澳门大学

李浩博、李思源、徐启昊、杨毅、陶威

主要联系人. Tel.: (+86) 13581975632; email: li.haobo@connect.um.edu.mo

## 摘要
AssistBCI是基于MetaBCI的脑机接口脑机协作系统与应用开发平台，为MetaBCI提供高效的技术部署方案，为新范式、新算法、新设备及新数据库提供了快速验证及应用的软件平台及中间件。为脑机接口领域的研究者、使用者和开发者提供了一个可视化的强大可靠工具，促进了脑机接口技术的快速迭代发展。
AssistBCI拟规范了脑机接口数据结构、通讯协议、同步器规范、预处理流程标准化，利用多进程和多线程，以及互自动连接，设备傻瓜化自动连接，运行连接快速，提高在线系统的实时效率。它能够实现对用户大脑意图的诱发、获取、分析和转换等全流程处理。不仅仅支持科研学术场景上范式调度，也支持将自由范式应用到真实自然场景中应用。

其主要功能特征包括：
1. 多传感器、设备支持
2.全透传刺激中间层（预构建和自定义设计）
3.多线程进程批量处理
4.优化动态实时处理和低延迟
5.图形化编程环境
7.事件同步支持
8.数据记录和在线离线分析

AssistBCI允许神经肌肉疾病患者根据使用场景独立设计视觉刺激布局，如SSVEP刺激位置 和大小，并支持自定义辅助操作，如鼠标操作。
另外，AssistBCI兼容MetaBCI平台设备、算法和范式。对于在MetaBCI验证的技术，无需额外开发，可通过AssistBCI直接适配。患者可快速应用最新BCI技术，提高使用乐趣，实现独立的电脑操作。对于新功能的实现，AssistBCI提供便捷开发工具，兼容MetaBCI中算法、设备和数据库，并提供可视化界面快速创建刺激界面。

  
![微信图片_20240804210208](https://github.com/user-attachments/assets/9302d6f1-28be-4762-b716-311a7cf1da9e)
    
<img width="416" alt="微信图片_20240804210214" src="https://github.com/user-attachments/assets/1dde5b67-bfd3-4a9d-9e0b-aa1b4040e90c">
  
<img width="410" alt="微信图片_20240804210202" src="https://github.com/user-attachments/assets/a6135eef-e6d5-4671-8a29-14dbb5fde770">
  
**关键词: 计算机辅助控制系统，二次开发平台, SSVEP, ms-eCCA, LinkMe, BlueBCIMetaBCI 创新应用开发赛项**
  
## 项目功能点明细表

 序号 | 功能点描述  | 量化指标
 ---- | ----- | ------  
 1  | 新增 Brainstim 中 SSVEP 范式刺激参数可视化设计工具 | 无 
 2  | 新增基于 MetaBCI 的二次开发框架 | 无  
 3  | 新增 Brainstim 中 SSVEP 指令尺寸独立设计 | 无 
 4  | 新增 Brainda 中 SSVEP 范式识别算法 | 2 种  
 5  | 新增 Brainflow 中采集数据实时保存方法 | 1 种 
 6  | 新增 Brainda 中采集数据便捷调用 | 1 种  
 7  | 新增 Brainflow 中设备支持 | 3 种 
 8  | 优化 Brainflow 中下载数据集存放位置 | 无  
 9  | 优化 Brainflow 中 worker 中 Brainda 算法的使用 | 无 

## 运行方法：

运行demos/brainstim_demos/stim_test.py即可

# 常见问题：

## 黑屏，显示问题：

### 可能原因：

psychopy版本不为1.5.27 

pyglet版本不为2022.1.4

多个显示器
