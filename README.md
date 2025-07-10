# AssistBCI-v2025——通用脑机接口辅助系统：基于 MetaBCI 的高效二次开发框架与全面个性化应用解决方案
澳门大学

李浩博、黄梓帆、朱峻毅、杨毅、陶威

主要联系人. Tel.: (+86) 13581975632; email: li.haobo@connect.um.edu.mo

## 1. 摘要
EEG的状态监测受限于诱发范式，存在标记被动、场景适应性不足等问题。对此，我们基于MetaBCI框架构建了"智能预测-用户反馈"的双向交互机制，通过无监督模型构建个性化状态特征库，实现了多维度状态的半自动标定与精准监测。系统采用轻量级交互设计：检测到显著状态变化时，发起交互验证/标定，显著降低标注成本和标签噪音。通过动态优化算法，自适应优化类内/类间距离，满足用户的个性化状态需求。系统面向智能家居调节、情绪管理、疼痛监测等领域，支持用户根据实际需求自定义状态标签，并设置智能设备联动规则（如空调温控、应急求助、焦虑缓解措施等）。随着用户的使用，个性化特征库将不断丰富，最终实现"感知-决策-执行"闭环的智能化服务。

**关键词: 特征提取，多维度生理与情感状态监测，个性化标注，轻量级交互，智能家居**

## 运行方法：

### 环境安装：

      pip install -r requirements.txt

### AssistBCI:
同时运行

    demos/brainflow_demos/AssistBCI_Backend_v2025.py
    demos/brainstim_demos/AssistBCI_interface.py
### EmoAdapt:
模型训练
            
      demos/EmoAdapt_train.py
模型常规线下测试
            
      demos/EmoAdapt_predict_offline.py
模型模拟线上测试

      demos/EmoAdapt_predict_online.py

## 2. 新增代码结构

   ```
   AssistBCI/
   │
   ├── README.md
   ├── requirements.txt
   │
   ├── assistbci_models/
   │   ├── classifier/                     # AssistBCI classifier storage location
   │   └── EmoAdapt/
   │       └── 0/
   │           └── model/
   │               └── Readme.txt          # Model download link, and model location
   │
   ├── demos/
   │   ├── EmoAdapt_experiment_train.py    # Training EmoAdapt on .edf files collected from NeuroDance
   │   ├── EmoAdapt_predict_offline.py     # Online simulation of proposed core Algorithm
   │   ├── EmoAdapt_predict_online.py      # Traditional offline test on extracted features by EmoAdapt
   │   ├── EmoAdapt_train.py               # Training EmoAdapt, the self-supervised model, based on SEED
   │   │
   │   ├── brainflow_demos/
   │   │   ├── AssistBCI_Backend_v2025.py
   │   │   ├── assistbci_worker_test.py    # Easy test of AssistBCI core Algorithm
   │   │   ├── Online_Emotion_experiment.py # Including experiment feedback and data saving
   │   │   ├── workers.py                  # AssistBCI Algorithms
   │   │   └── device_worker.py            # Device and worker manager
   │   │
   │   └── brainstim_demos/
   │       ├── libvlc.dll                  # Requirement of VLC
   │       ├── stim_demo.py                # Adding Emotion experiment demo
   │       ├── AssistBCI_interface.py      # Main interface for AssistBCI
   │       └── light_virtual_trigger_test.py  # Test file for light/virtual trigger
   │
   ├── metabci/
   │   ├── utils/
   │   │   ├── mijia_action.py             # A easy API for controlling mijia device by customized shortcut
   │   │   ├── mijia_connect_home.py       # A visual interface for creating miji customized shortcut
   │   │   ├── sharedmemory.py             # Easy tool for system data/flags storage and transmission
   │   │   ├── sharedmemory_ManageTool.py  # Shared Memory management tool
   │   │   └── __init__.py
   │   │
   │   ├── brainda/
   │   │   ├── algorithms/
   │   │   │   └── self_supervised_learning/
   │   │   │       ├── Base.py             # TorchDataset for multiple arrays input (Eg.(x,y, aug_x, aug_y)), NTXentLoss, 1d/2d sin cos pos embed
   │   │   │       ├── EmoAdapt.py         # A self-supervised model
   │   │   │       ├── prototype.py        # Online learning model
   │   │   │       └── utils.py            # Augment data methods, t-SNE visualization
   │   │   │
   │   │   ├── datasets/
   │   │   │   ├── seed.py                 # Read EEG data in seed dataset
   │   │   │   ├── unlabled_eeg.py         # Read unlabeld edf files (tested on .edf file saved by NeuroDance)
   │   │   │   └── __init__.py
   │   │   │
   │   │   ├── paradigms/
   │   │   │   └── emotion.py              # Adding emotion paradigms
   │   │   │
   │   │   └── utils/
   │   │       └── download.py             # Fix path bug under windows when local dataset locate at different disc (Eg. code/project in E:// while dataset in D://)
   │   │
   │   ├── brainflow/
   │   │   └── amplifiers.py               # Enhance ringbuffer, adding data saving function in Marker, fix BaseAmplifier.up_worker, adding Devices: NeuroDance, BlueBCI, add auto-data-saving in BaseAmplifier after release the worker
   │   │
   │   └── brainstim/
   │       ├── framework.py                # Adding allowGUI control (修复)
   │       ├── paradigm.py                 # Adding light and virtual tigger support, adding emotion paradim
   │       └── utils.py                    # Adding light and virtual trigger support
   │
   └── vlc/
   ```

## 3. 新增功能

| # | Feature Description | Subplatform | Code Path | Classes/Functions |
|---|----------------------|-------------|-----------|-------------------|
| 1 | Added BlueBCI/NeuroDance device support | Brainflow | `metabci/brainflow/amplifiers.py` | 1. `BlueBCI()`<br>2. `NeuroDance()` |
| 2 | Added Marker data saving | Brainflow | `metabci/brainflow/amplifiers.py` | `Marker()` |
| 3 | Added auto Marker data saving | Brainflow | `metabci/brainflow/amplifiers.py` | `BaseAmplifier.unregister_worker()` (partial) |
| 4 | Added Emotion/physiological paradigm | Brainstim | `metabci/brainstim/paradigm.py` | 1. `Emotion()`<br>2. `paradigm()` (partial) |
| 5 | Added phototube/timestamp marking | Brainstim | `metabci/brainstim/utils.py` | 1. `Virtual_trigger()`<br>2. `Light_trigger()` |
| 6 | Adapted phototube/timestamp marking | Brainstim | `metabci/brainstim/paradigm.py` | `paradigm()` (partial) |
| 7 | Added Emotion paradigm | Brainda | `metabci/brainda/paradigms/emotion.py` | `Emotion()` |
| 8 | Added SEED dataset (requires offline data) | Brainda | `metabci/brainda/datasets/seed.py` | `SEED()` |
| 9 | Added self-supervised model EmoAdapt | Brainda | `metabci/brainda/algorithms/self_supervised_learning/EmoAdapt.py` | `EmoAdapt()` |
| 10 | Added augmentation methods, t-SNE visualization | Brainda | `metabci/brainda/algorithms/self_supervised_learning/utils.py` | 1. `augment_data()`<br>2. `plot_embedding()` |
| 11 | Added self-supervised tools: TorchDataset (multi-array input), NTXentLoss, 1d/2d sin cos position encoding | Brainda | `metabci/brainda/algorithms/self_supervised_learning/Base.py` | 1. `TorchDataset()`<br>2. `NTXentLoss()`<br>3. `get_1d_sincos_pos_embed()`<br>4. `get_2d_sincos_pos_embed()` |
| 12 | Added unlabeled .edf file reading | Brainda | `metabci/brainda/datasets/unlabled_eeg.py` | `unLabeled_EEG()` |
| 13 | Added online adaptive classifier | Brainda | `metabci/brainda/algorithms/self_supervised_learning/prototype.py` | 1. `PCA_classifier()`<br>2. `PCA_EmoAdapt_online()` |
| 14 | Added Mi Home device control | General | `metabci/utils/mijia_action.py` | `execute_state_actions()` |
| 15 | Added visual Mi Home shortcut setup | General | `metabci/utils/mijia_connect_home.py` | `Mijia_home()` |
| 16 | Added Mmap-based shared memory | General | `metabci/utils/sharedmemory.py` | `SharedDict()` |
| 17 | Added shared memory visual management tool | General | `metabci/utils/sharedmemory_ManageTool.py` | `SharedMemoryViewer()` |

### 功能测试

1. **Device Testing**  
   Test results: Stable data reading, marking delay <1ms

5-6. **Test File**: `demos/brainstim_demos/light_virtual_trigger_test.py`  
   Results: Stable operation, virtual trigger sync speed near memory speed, light trigger sync stable

7-11. **Test File**: `demos/EmoAdapt_predict_online.py`  
   Notes: Provides 4/8 channel models - check model path and data channels in `get_args()`  
   Results: Correct data reading, good t-SNE visualization, 92.5% cross-session accuracy after self-supervised training (4-channel)

12. **Test File**: `demos/EmoAdapt_experiment_train.py`  
   Results: Normal data reading

13. **PCA_classifier() Test File**: `demos/EmoAdapt_predict_online.py`  
   Results: Works normally with high prediction accuracy

13. **PCA_EmoAdapt_classifier() Test File**: `demos/brainflow_demos/assistbci_worker_test.py`  
   Notes: Use ≥8 channel model/device, adjust channel mapping in `workers.py` for best performance  
   Results: Algorithm works normally

14-15. **Test Files**: Run corresponding files (run 15 first to setup shortcuts)  
   Notes: Test data pre-saved in `assistbci_models/classifier/` - delete manually before testing to enable neutral state pre-training  
   Results: Normal Mi Home login, state reading, shortcut creation, device control

16-17. **Test Files**: Run corresponding files  
   Results: Normal read/write, memory expansion works, visual management functions work

## 4. 修改与优化

| # | Fix Description | Subplatform | Code Path | Classes/Functions |
|---|------------------|-------------|-----------|-------------------|
| 1 | Fixed Windows path issue when data/program on different drives | Brainda | `metabci/brainda/utils/download.py` | `_get_file()` |
| 2 | Improved Ringbuffer functionality | Brainflow | `metabci/brainflow/amplifiers.py` | `RingBuffer.isEmpty()` |
| 3 | Fixed BaseAmplifier.up_worker() naming issue (all workers named "feedback_worker") | Brainflow | `metabci/brainflow/amplifiers.py` | `BaseAmplifier.up_worker()` |
| 4 | Enhanced Experiment GUI window control | Brainstim | `metabci/brainstim/framework.py` | `Experiment.get_window()` |
| 5 | Improved device/algorithm encapsulation | Brainflow | `metabci/brainflow_demos/device_worker.py` | `Device()` |

### 修改与优化测试

1. **Test Case**:  
   Place dataset and project on different Windows drives - error occurs with modification commented out, works normally when uncommented  
   Error screenshot (cause: src_file missing drive path)

2-4. No testing needed (simple modifications)

5. **Test Program**: `demos/brainflow_demos/AssistBCI_Backend_v2025.py`  
   Notes: Adjust default config based on devices/algorithms, or modify via shared memory management tool after startup

