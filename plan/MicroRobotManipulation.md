# 微纳机器人操作 - 视频轨迹生成与动作预测

## 1. 项目概述

### 1.1 项目目标

构建一个**语言条件化的微纳操作视频轨迹生成与动作预测模型**，能够：
- 给定当前显微镜图像 + 语言指令，生成未来操作轨迹视频
- 同时预测对应的操作动作序列（stage XY + pipette XYZ）
- 支持多种操作任务（细胞移动、注射、吸取等）
- 支持静态和动态目标（非游动精子、游动精子等）
- 通过语言指令指定目标类型和操作意图，实现泛化

### 1.2 核心需求

| 需求维度 | 具体要求 |
|---------|---------|
| **输入** | 显微镜2D RGB图像 (1600x1200) + 语言指令 + 可选历史帧 |
| **输出** | 预测视频帧序列 + 动作序列（stage XY, pipette XYZ） |
| **泛化性** | 语言条件化 → 不同目标、不同操作、不同位置 |
| **硬件约束** | 6-8x NVIDIA 5090 GPU（单卡32GB VRAM） |
| **实时性** | 训练离线；推理需要一定实时性以支持闭环控制 |

### 1.3 操作场景矩阵

```
┌─────────────┬──────────────┬──────────────┐
│ 目标类型     │ 静态细胞      │ 动态细胞      │
│             │ (非游动精子等) │ (游动精子等)   │
├─────────────┼──────────────┼──────────────┤
│ 简单操作     │ 移动到目标位置 │ Stage追踪定位  │
│ (Stage控制)  │              │              │
├─────────────┼──────────────┼──────────────┤
│ 复杂操作     │ 细胞注射      │ 尾部制动       │
│ (Pipette控制)│ 细胞吸取      │ 精子吸取       │
│             │ 细胞移动      │ 精子注射(ICSI) │
└─────────────┴──────────────┴──────────────┘
```

---

## 2. 系统架构设计

### 2.1 硬件系统组成

```
显微镜系统
├── 相机 → 2D RGB图像/视频 (输入, 1600x1200)
├── 电动载物台 Stage → XY平面移动 (控制目标)
└── 微操作器 Pipette → XYZ 3-DOF (控制目标)
    └── 显微注射针 → 视野中可见tip
```

### 2.2 坐标系定义

- **图像坐标系**：像素 (u, v)，原点左上角
- **Stage坐标系**：微米 (x, y)，通过 um/pixel 转换矩阵关联
- **Pipette坐标系**：微米 (x, y, z)，z为深度/聚焦方向

**关键转换**：
- Stage XY ↔ 图像像素：通过标定的 um/pixel 系数转换
- Pipette Z：通过图像清晰度（焦距）推断深度关系

### 2.3 深度信息处理策略

由于没有3D传感器，深度信息通过以下方式获取：

| 方案 | 方法 | 适用场景 |
|------|------|---------|
| **主要方案** | 焦距推断（图像清晰度） | 判断pipette和细胞是否在同一深度 |
| **辅助方案** | 多焦平面采集（focus stack） | 确定最佳焦平面位置 |
| **可选增强** | 深度估计网络（Marigold/DepthAnything） | 提供连续深度信息 |

---

## 3. 数据采集设计

### 3.1 数据采集系统架构

```
数据采集系统
├── 硬件接口层
│   ├── CameraDriver        # 相机驱动 (OpenCV/Micro-Manager)
│   ├── StageController     # 电动台控制 (Prior/ASI)
│   └── PipetteController   # 微操作器控制 (Eppendorf/Sutter)
│
├── 同步采集层
│   ├── FrameSynchronizer   # 多线程帧同步器
│   └── DataRecorder        # 结构化数据记录器
│
├── 标定系统
│   ├── PixelToUmCalibrator # 像素-微米标定
│   └── FocusCalibrator     # 焦距-深度标定
│
└── 标注工具
    ├── TaskAnnotator       # 任务级标注
    └── SubgoalAnnotator    # 子目标级标注
```

### 3.2 硬件接口设计

**相机驱动**：
- 支持两种接口：OpenCV VideoCapture（通用USB相机）和 Micro-Manager（专业显微镜软件）
- 功能：连接相机、设置分辨率/帧率/曝光、捕获图像、获取时间戳
- 关键参数：分辨率1600x1200，帧率30fps

**电动台控制器**：
- 支持Prior Scientific和ASI两种品牌
- 通信协议：Serial (RS-232/USB-Serial)
- 功能：连接设备、读取当前位置、相对/绝对移动、等待移动完成
- 关键命令：查询位置、相对移动、绝对移动

**微操作器控制器**：
- 支持Eppendorf InjectMan和Sutter MP-285两种品牌
- 功能：连接设备、读取XYZ位置、相对/绝对移动
- 关键：Z轴位置用于深度信息

### 3.3 同步采集设计

**帧同步器（FrameSynchronizer）**：
- 使用独立线程分别读取相机、电动台、微操作器
- 通过时间戳对齐不同设备的数据
- 同步精度要求：<50ms（对于30fps足够）

**工作流程**：
1. 相机线程：持续捕获图像，放入图像队列
2. 电动台线程：100Hz读取位置，放入位置队列
3. 操作器线程：100Hz读取位置，放入位置队列
4. 同步线程：从队列取数据，按时间戳匹配，输出同步帧

**数据记录器（DataRecorder）**：
- 记录每帧数据：图像、时间戳、Stage XY、Pipette XYZ、清晰度指标
- 自动生成相对动作（delta）：从绝对位置计算帧间变化
- 保存为结构化JSON格式

### 3.4 标定系统设计

**像素-微米标定（PixelToUmCalibrator）**：
- 方法1：使用标准刻度尺（graticule），自动检测刻度线间距
- 方法2：手动选择两个已知距离的点
- 输出：um_per_pixel 系数（X/Y方向）

**多物镜标定**：
- 显微镜有多个物镜（4x, 10x, 20x, 40x），每个物镜的um/pixel不同
- 需要分别标定并保存

**焦距-深度标定（FocusDepthCalibrator）**：
- 采集不同Z位置的图像序列（Z-stack）
- 计算每个位置的清晰度（拉普拉斯方差）
- 拟合焦距-深度曲线
- 用于判断pipette和细胞是否在同一焦平面

### 3.5 语言标注设计

**标注粒度层级**：

| 层级 | 示例 | 用途 |
|------|------|------|
| Level 1 - 任务级 | "对游动精子进行ICSI操作" | 高层任务描述 |
| Level 2 - 阶段级 | "将pipette移动到精子尾部上方" | 分步子目标 |
| Level 3 - 帧级 | (可选) | 精细学习 |

**标注模板**：
```yaml
task_template:
  target_type: "motile_sperm"
  operation: "ICSI"
  goal_description: "将pipette移动到精子尾部，制动后吸取，然后注射到卵细胞"
  sub_goals:
    - "stage移动，将精子控制在视野中央"
    - "pipette下移到精子尾部深度"
    - "pipette水平移动到精子尾部"
    - "制动精子尾部"
    - "吸取精子"
    - "移动到卵细胞附近"
    - "注射精子"
```

### 3.6 数据规模规划

| 阶段 | 目标 | 数据量 | 优先级 |
|------|------|-------|--------|
| 阶段1 | 基础移动（stage控制） | ~500 episodes | 高 |
| 阶段2 | 静态细胞操作 | ~300 episodes | 高 |
| 阶段3 | 动态细胞追踪 | ~300 episodes | 高 |
| 阶段4 | 完整ICSI流程 | ~200 episodes | 中 |
| **总计** | | **~1300 episodes** | |

每个episode长度：10-60秒，约300-1800帧（30fps）

### 3.7 数据格式设计

**目录结构**：
```
dataset/
├── metadata.json
├── episodes/
│   ├── episode_0001/
│   │   ├── frames/           # 图像帧 (000000.png, 000001.png, ...)
│   │   ├── actions.json      # 相对动作序列
│   │   ├── states.json       # 绝对状态序列
│   │   └── metadata.json     # episode元数据
│   └── ...
├── language_annotations/
│   ├── task_descriptions.jsonl
│   └── subgoal_annotations.jsonl
└── calibration/
    ├── pixel_to_um.json
    └── camera_intrinsics.json
```

**actions.json格式**：
```json
{
  "episode_id": "episode_0001",
  "fps": 30,
  "frames": [
    {
      "frame_idx": 1,
      "timestamp": 0.033,
      "stage_delta_xy": [1.5, -0.3],
      "pipette_delta_xyz": [0.0, 0.0, 0.0]
    }
  ]
}
```

**states.json格式**：
```json
{
  "episode_id": "episode_0001",
  "frames": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "stage_xy": [100.5, 200.3],
      "pipette_xyz": [50.2, 80.1, 10.5],
      "sharpness_cell": 0.85,
      "sharpness_pipette": 0.72,
      "target_in_fov": true
    }
  ]
}
```

### 3.8 精度要求分析

**精子操作的精度需求**：

| 分辨率 | 精子大小 | 2px精度 | 可行性 |
|--------|---------|---------|--------|
| 1600x1200 (原图) | 19-30 px | 1.2 um | ✓ 最佳 |
| 1024x768 | 12-19 px | 1.2 um | ✓ 可接受 |
| 512x384 | 6-10 px | 1.2 um | ✗ 太小 |

**结论**：动作预测必须使用 ≥1024 分辨率的特征

### 3.9 数据预处理流程

1. **动作表示转换**：绝对位置 → 相对变化量（delta）
2. **清晰度计算**：拉普拉斯方差、梯度幅值
3. **图像预处理**：
   - 保存原始1600x1200（用于高分辨率编码）
   - 生成下采样512x384（用于低分辨率编码）
   - 可选：中心裁剪1024x1024

### 3.10 数据采集脚本设计

**主采集器（MicroManipulationCollector）**：
- 整合所有硬件组件
- 提供标定模式：交互式选择点进行像素-微米标定
- 提供采集模式：实时预览 + 按键控制录制

**操作流程**：
1. 启动程序，初始化硬件
2. 标定模式：放置刻度尺，选择两个点，输入已知距离
3. 采集模式：
   - 实时显示图像、Stage位置、Pipette位置
   - 按SPACE开始/停止录制
   - 按S保存当前episode
   - 按D删除当前episode
   - 按Q退出

**配置文件**：
```yaml
# collector_config.yaml
output_dir: "./data/collected"
camera:
  type: "opencv"
  resolution: [1600, 1200]
  fps: 30
stage:
  type: "prior"
  port: "COM3"
pipette:
  type: "eppendorf"
  port: "COM4"
calibration_file: "./config/calibration.yaml"
show_preview: true
```

---

## 4. 模型架构设计

### 4.1 整体架构：双分辨率双分支

**核心设计**：编码用高分辨率保留小目标，生成用低分辨率节省显存

```
原始 1600x1200 输入帧
        │
        ├──→ [高分辨率编码路径] 1600x1200
        │    ├── InternViT-6B (12 tiles, 3072 tokens)
        │    ├── 保留精子(19-30px)和pipette tip细节
        │    └── 用于动作预测
        │
        └──→ [低分辨率编码路径] 512x384
             ├── InternViT-6B 或 SigLIP (1 tile, 256 tokens)
             └── 用于视频预测
```

### 4.2 视觉编码器选择

**双编码器方案**：

| 路径 | 编码器 | 输入分辨率 | 输出tokens | 用途 |
|------|--------|-----------|-----------|------|
| 高分辨率 | InternViT-6B + LoRA | 1600x1200 (12 tiles) | 3072 | 动作预测 |
| 低分辨率 | InternViT-6B (共享) 或 SigLIP | 512x384 (1 tile) | 256 | 视频预测 |

**为什么需要双分辨率**：
- 视频预测：只需预测大致场景趋势，512x384足够
- 动作预测：需要精确到19-30px的精子位置，必须用高分辨率

### 4.3 视频生成分支（低分辨率）

- 基础模型：CogVideoX-5B + LoRA (rank=64)
- 输入：256 tokens + language embedding
- 输出：16帧 × 512x384 RGB
- 生成分辨率：512x384（平衡质量和显存）

### 4.4 动作预测分支（高分辨率）

- 输入：3072 tokens（高分辨率特征）+ language embedding
- 输出：16步 × 5维动作序列
- 动作空间：stage_dx, stage_dy, pipette_dx, pipette_dy, pipette_dz
- 预测头：Diffusion Action Head（建模多模态分布）

### 4.5 条件注入机制

- 语言条件：Cross-Attention（语言特征作为Key/Value）
- 动作条件：可选的历史动作编码

### 4.6 计算成本分析

**显存估算（batch_size=2）**：
- 高分辨率路径：~16GB (gradient_checkpointing)
- 低分辨率路径：~12GB
- 总计：~28GB ✓ 适合5090 32GB

**训练速度**：
- 每iteration：~1.25秒
- 30,000 iterations × 6卡：~3-4小时

---

## 5. 训练策略

### 5.1 多阶段训练流程

```
阶段0: 数据采集与标注 (2周)
    ↓
阶段1: 视频预测预训练 (无动作标签)
    │  - 大量显微镜视频
    │  - 学习显微镜域的视觉动态
    ↓
阶段2: 动作条件化联合训练 (有动作标签)
    │  - 联合训练视频+动作
    │  - 学习动作-视觉对应关系
    ↓
阶段3: 语言条件化微调
    │  - 任务级描述作为条件
    │  - 子目标级描述分步训练
    ↓
阶段4: 可选的RL后训练
       - 在线微调优化成功率
```

### 5.2 阶段1：视频预测预训练

- 数据：所有采集的视频数据（不需要动作标签）
- 模型：CogVideoX-5B + LoRA (rank=64)
- 分辨率：512x384
- 帧数：16帧
- 损失：L1 + LPIPS + SSIM
- 学习率：1e-4，cosine schedule
- 训练步数：50,000

### 5.3 阶段2：动作条件化联合训练

- 数据：有动作标签的数据子集
- 模型：预训练视频预测器 + 动作预测头
- 冻结视觉编码器，训练动作头
- 损失：video_loss + 0.5 × action_loss
- 学习率：5e-5
- 训练步数：30,000

### 5.4 阶段3：语言条件化微调

- 数据：带有语言标注的数据
- 策略：课程学习
  - Stage 1：简单任务（stage_move, pipette_move）
  - Stage 2：中等任务（cell_injection, cell_aspiration）
  - Stage 3：复杂任务（ICSI_full, motile_sperm_tracking）

### 5.5 显存优化策略

- 梯度检查点 (gradient_checkpointing)
- 混合精度训练 (bf16)
- LoRA微调 (rank=64)
- 梯度累积 (accumulation_steps=8)
- DeepSpeed ZeRO-2 分布式训练

---

## 6. 评估体系

### 6.1 视频预测评估

| 指标 | 说明 | 目标值 |
|------|------|--------|
| FVD | 视频生成质量 | < 200 |
| LPIPS | 感知相似度 | < 0.2 |
| SSIM | 结构相似性 | > 0.85 |

### 6.2 动作预测评估

| 指标 | 说明 | 目标值 |
|------|------|--------|
| MSE | 动作误差 | < 0.01 (归一化) |
| 成功率 | 任务完成率 | > 70% |
| 碰撞率 | pipette碰撞率 | < 5% |

### 6.3 仿真评估环境

建议构建仿真环境进行快速迭代：
- 模拟stage XY移动
- 模拟pipette XYZ移动
- 模拟细胞运动（静态/动态）
- 计算奖励（距离、对齐、任务完成）

---

## 7. 推理与部署

### 7.1 推理流程

1. 采集当前帧 (1600x1200)
2. 高分辨率编码 → 动作特征
3. 低分辨率编码 → 视频特征
4. 预测未来16步动作序列
5. 预测未来16帧视频
6. 执行第1步动作
7. 采集新帧，重新预测（MPC风格）

### 7.2 闭环控制策略

- 控制频率：5-10Hz（受限于模型推理~100-200ms）
- 策略：每次预测16步，只执行第1步，然后重新预测
- 优势：能够适应动态变化（如游动精子）

---

## 8. 实施计划

### 8.1 阶段规划

| 阶段 | 时间 | 目标 | 产出 |
|------|------|------|------|
| **Phase 0** | 2周 | 系统搭建 | 数据采集系统、标定工具 |
| **Phase 1** | 4周 | 数据采集 | ~500 episodes基础数据 |
| **Phase 2** | 3周 | 基础模型 | 视频预测预训练模型 |
| **Phase 3** | 3周 | 动作预测 | 联合训练模型 |
| **Phase 4** | 3周 | 语言条件化 | 完整系统 |
| **Phase 5** | 2周 | 评估优化 | 仿真+真实评估 |
| **总计** | ~17周 | | |

### 8.2 Phase 0：系统搭建（第1-2周）

**任务清单**：
- 硬件标定：相机内参、Stage XY um/pixel、Pipette XYZ um/pixel、焦距-深度关系
- 数据采集脚本：相机采集、位置读取、数据同步保存
- 基础框架：项目结构、配置系统、日志系统

### 8.3 Phase 1：数据采集（第3-6周）

- Week 3：基础移动数据（Stage移动到指定位置）- 200 episodes
- Week 4：简单操作数据（Pipette接近目标）- 150 episodes
- Week 5：复杂操作数据（细胞注射/吸取）- 100 episodes
- Week 6：动态目标数据（游动精子追踪）- 100 episodes

### 8.4 Phase 2-3：模型开发（第7-12周）

- Week 7-8：视频预测预训练（数据pipeline + CogVideoX训练）
- Week 9-10：动作预测头（架构实现 + 联合训练）
- Week 11-12：模型优化（超参数调优、显存优化）

### 8.5 Phase 4-5：系统集成（第13-17周）

- Week 13-14：语言条件化（语言编码器 + 条件化训练）
- Week 15-16：评估系统（仿真环境 + 自动评估）
- Week 17：系统集成（推理pipeline + 真实环境测试）

---

## 9. 关键技术风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| 显微镜图像域gap大 | 预训练模型迁移困难 | 大量域内数据预训练；LoRA微调 |
| 深度信息不准确 | Z轴动作预测差 | 多焦平面采集；深度估计网络辅助 |
| 动作-视觉对齐差 | 动作预测不准 | 更好的特征融合；增加训练数据 |
| 显存不足 | 无法训练大模型 | 降低分辨率；减小LoRA rank；梯度累积 |
| 泛化性不足 | 新任务表现差 | 语言条件化；课程学习；数据增强 |

---

## 10. 参考资源

### 10.1 相关项目

| 项目 | 参考价值 |
|------|---------|
| GR00T-Dreams | 视频世界模型+动作预测架构 |
| TesserAct | 4D世界模型，RGB-DN表示 |
| ViDAR | 视觉点云预测，自监督预训练 |

### 10.2 基础模型

| 模型 | 用途 |
|------|------|
| CogVideoX-5B | 视频生成基座 |
| InternVL2-8B | 视觉编码器 |
| SigLIP-SO400M | 轻量视觉编码器 |
| DepthAnything-V2 | 深度估计 |

### 10.3 工具库

| 工具 | 用途 |
|------|------|
| PyTorch + DeepSpeed | 训练框架 |
| Transformers | 模型加载 |
| PEFT (LoRA) | 参数高效微调 |
| OpenCV | 图像处理 |

---

## 附录A：项目目录结构

```
MicroRobotManipulation/
├── config/
│   ├── collector_config.yaml   # 数据采集配置
│   ├── calibration.yaml        # 标定参数
│   ├── pretrain.yaml           # 预训练配置
│   └── finetune.yaml           # 微调配置
├── data/
│   ├── collector/              # 数据采集脚本
│   │   ├── camera_driver.py
│   │   ├── stage_controller.py
│   │   ├── pipette_controller.py
│   │   ├── synchronizer.py
│   │   ├── data_recorder.py
│   │   └── main_collector.py
│   ├── calibration/            # 标定工具
│   │   ├── pixel_to_um.py
│   │   └── focus_depth.py
│   ├── preprocessor/           # 数据预处理
│   └── dataset/                # 数据集
├── models/
│   ├── visual_encoder.py
│   ├── video_predictor.py
│   ├── action_head.py
│   └── microbot_dreamer.py
├── training/
│   ├── pretrain.py
│   ├── finetune.py
│   └── utils.py
├── evaluation/
│   ├── metrics.py
│   ├── simulator.py
│   └── evaluator.py
├── inference/
│   ├── predictor.py
│   └── controller.py
├── scripts/
│   ├── collect_data.py
│   └── visualize.py
└── plan/
    └── MicroRobotManipulation.md  # 本文件
```
