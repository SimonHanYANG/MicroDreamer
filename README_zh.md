# MicroDreamer

**面向微纳机器人操作的语言条件化视频轨迹生成与动作预测**

[English](./README.md) | 中文

MicroDreamer 是一个双输出模型，能够同时生成未来视频帧并预测微纳机器人在显微镜下的操作动作序列。该模型专为细胞操作任务设计，如 ICSI（卵胞浆内单精子注射）、细胞分选和胚胎移植等。

## 核心特性

- **双分辨率架构**：高分辨率路径（1600×1200）用于动作预测，低分辨率路径（512×384）用于视频预测
- **语言条件化**：支持自然语言指令描述任务，通过 Flan-T5 编码器注入交叉注意力
- **扩散动作预测**：基于 DDPM 的 5-DOF 动作序列预测（stage_dx, stage_dy, pip_dx, pip_dy, pip_dz）
- **LoRA 视频生成**：CogVideoX 风格的时序变换器，LoRA 微调（约 10% 可训练参数）
- **硬件抽象层**：支持虚拟设备测试和真实硬件（Basler 相机、Nikon Ti2E 载物台、HTTP API 移液器）

## 系统架构

```
输入帧 (1600×1200)
        │
        ├──→ 高分辨率路径 (3072 tokens)
        │    └──→ TileAttentionEncoder (12 tiles × 448×448)
        │         └──→ DiffusionActionHead → 动作序列 (T, 5)
        │
        └──→ 低分辨率路径 (256 tokens)
             └──→ VideoPredictionModel (时序变换器 + LoRA)
                  └──→ 预测视频帧 (T, C, H, W)

语言指令 → Flan-T5 编码器 → 交叉注意力注入
```

详细架构说明请参阅 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)。

## 项目结构

```
MicroDreamer/
├── config/           # 配置管理（YAML）
├── hardware/         # 硬件抽象层
│   ├── virtual/      # 虚拟设备（测试用）
│   ├── camera/       # Basler 相机驱动
│   ├── stage/        # Nikon Ti2E 载物台控制器
│   └── pipette/      # HTTP API 移液器控制器
├── data/             # 数据采集与预处理
│   ├── collector/    # 多设备同步数据记录器
│   ├── preprocessor/ # 帧/动作预处理
│   ├── annotation/   # 语言标注格式
│   └── dataset.py    # PyTorch Dataset
├── models/           # 模型组件
│   ├── action/       # 动作预测（视觉编码器 + 扩散头）
│   ├── video/        # 视频预测（CogVideoX + LoRA）
│   └── language/     # 语言编码器
├── inference/        # 推理流水线
├── scripts/          # 训练/评估/标定脚本
├── tests/            # 单元测试与集成测试
├── docs/             # 开发文档
└── utils/            # 工具（日志、标定）
```

## 快速开始

```bash
# 激活 conda 环境
conda activate microdreamer

# 安装依赖
pip install -r requirements.txt

# 运行全部测试
python tests/run_all_tests.py

# 启动数据采集 UI（推荐）
python scripts/collect_ui.py

# 或通过命令行采集虚拟数据
python scripts/collect_data.py --mode virtual --num_episodes 5 --output_dir ./data/raw

# 训练动作预测模型
python scripts/train_action.py --data_dir ./data/raw --output_dir ./outputs --simple_lang

# 训练视频预测模型
python scripts/train_video.py --data_dir ./data/raw --output_dir ./outputs

# 评估模型
python scripts/evaluate.py --data_dir ./data/raw --action_ckpt ./outputs/checkpoints/action_best.pt

# 运行推理演示
python inference/predict.py --demo

# 运行标定
python scripts/calibrate.py --mode all
```

### 数据采集 UI

项目包含基于 tkinter 的交互式数据采集 GUI：

```bash
python scripts/collect_ui.py
```

功能特性：
- **实时相机预览**：实时显示显微镜画面
- **Stage 控制**：手动 XY 移动，可配置步长
- **Pipette 控制**：XY 移动 + Z 轴升降
- **PID 自动定位**：点击画面设置目标位置，PID 控制器自动移动载物台
- **数据录制**：开始/停止录制，输入任务描述
- **Episode 管理**：浏览已采集的数据片段

详细使用说明请参阅 [数据采集指南](docs/DATA_COLLECTION_GUIDE.md)。

## 文档

- [系统架构](docs/ARCHITECTURE.md) — 架构设计与模块说明
- [API 参考](docs/API.md) — 完整 API 文档
- [数据采集指南](docs/DATA_COLLECTION_GUIDE.md) — 详细的采集流程与标注规范
- [Data Collection Guide (EN)](docs/DATA_COLLECTION_GUIDE_EN.md) — 英文版
- [端到端测试手册](docs/E2E_TEST_GUIDE.md) — 完整测试流程与操作步骤
- [E2E Test Guide (EN)](docs/E2E_TEST_GUIDE_EN.md) — English version
- [开发日志](docs/DEVELOPMENT.md) — 开发进度与更新记录

## 硬件要求

| 用途 | 配置 |
|------|------|
| 训练 | 6-8× NVIDIA 5090 GPU |
| 推理 | 1-2× GPU |
| 测试 | 仅 CPU（虚拟设备） |

## 训练功能

- TensorBoard 日志（损失曲线、学习率、预测可视化）
- 梯度裁剪（max_norm=1.0）
- 余弦学习率调度器 + 线性预热
- 断点续训（保存/恢复完整优化器状态）
- 早停机制（可配置耐心值）
- 混合精度训练（FP16）
- LoRA 参数组（5 倍学习率）

## 评估指标

**动作预测**：MSE、MAE、逐维度 MSE、终点误差、轨迹长度、一致性

**视频预测**：像素 MSE/MAE、PSNR、SSIM、时序一致性、FVD

## 数据格式

每个 episode 包含：
- `data.npz`：帧序列、载物台位置、移液器位置、时间戳
- `metadata.json`：任务描述、子目标列表、帧数

## 端到端测试命令

```bash
# 0. 环境准备
conda activate microdreamer
cd D:\SimonWorkspace\MicroRobotDataGen\MicroDreamer

# 1. 生成 mock 数据（含目标点）
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160

# 1.5 可视化检查数据（交互式 GUI）
python scripts/viz_mock_data.py

# 2. 运行全部单元测试（~34 tests）
python tests/run_all_tests.py

# 3. 训练动作模型（3 epochs，测试配置）
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3

# 4. 训练视频模型（3 epochs，测试配置）
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3

# 5. 评估两个模型
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml

# 5.5 可视化评估结果
python scripts/viz_mock_data.py --data_dir ./data/test_raw

# 6. 推理
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"

# 6.5 可视化推理结果
python scripts/viz_mock_data.py --data_dir ./data/test_raw

# 7. 清理
rmdir /s /q data\test_raw
rmdir /s /q outputs\test
```

快速验证（不需要 checkpoint）：
```bash
python inference/predict.py --demo --device cpu
```

静态可视化（生成 PNG）：
```bash
python scripts/visualize_mock_data.py --output_dir ./data/viz_mock --save_dir ./outputs/viz
```

详见 [端到端测试手册](docs/E2E_TEST_GUIDE.md)。

## 许可证

待定
