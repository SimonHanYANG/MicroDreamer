# MicroDreamer 端到端测试手册

## 概述

本手册提供完整的端到端测试流程，验证 MicroDreamer 系统的所有核心功能：

1. **数据生成** — 生成模拟训练数据
2. **数据可视化** — 交互式检查生成的数据质量
3. **数据加载** — 验证 Dataset 加载和预处理
4. **动作模型训练** — 训练动作预测模型
5. **视频模型训练** — 训练视频预测模型
6. **模型评估** — 评估两个模型的指标
7. **推理预测** — 使用训练好的模型进行推理

> 可视化工具 `viz_mock_data.py` 可在数据生成后、训练前、评估后、推理后任意阶段使用，用于检查数据质量和模型输出。

---

## 环境准备

### 1. 激活 conda 环境

```cmd
conda activate microdreamer
```

### 2. 进入项目目录

```cmd
cd D:\SimonWorkspace\MicroRobotDataGen\MicroDreamer
```

### 3. 验证环境

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 测试流程

### 步骤 1：生成模拟测试数据

生成 10 个 episode 的模拟数据，用于后续所有测试。

**CMD:**
```cmd
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160
```

**Bash:**
```bash
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160
```

**验证结果：**
- 应看到 `Done! Generated 10 episodes in data\test_raw`
- 验证数据格式：
  ```cmd
  python -c "import numpy as np; d=np.load('./data/test_raw/episode_20260614_000000_0000/data.npz'); print('frames:', d['frames'].shape, 'stage:', d['stage_positions'].shape, 'pipette:', d['pipette_positions'].shape)"
  ```
- 预期输出：`frames: (50, 160, 200) stage: (50, 2) pipette: (50, 3)`

---

### 步骤 1.5：可视化检查生成的数据（推荐）

使用交互式可视化工具检查生成的 mock 数据是否合理。

**交互式 GUI（推荐）：**
```cmd
python scripts/viz_mock_data.py
```

**功能：**
- 下拉切换 episode，逐帧播放显微镜画面
- 查看 Stage XY 轨迹图（按子目标颜色编码）
- 查看 Stage/Pipette 位置随时间变化
- 查看 5-DOF action deltas
- 查看子目标时间线、帧统计信息
- 所有图表随帧滑块同步联动

**检查要点：**
- 帧中有可辨识的运动物体（白色圆圈）
- Stage 轨迹为连续平滑路径
- Pipette Z 轴有下降趋势（模拟吸液）
- 子目标时间段划分合理

**静态可视化（生成 PNG）：**
```cmd
python scripts/visualize_mock_data.py --output_dir ./data/viz_mock --save_dir ./outputs/viz
```
在 `./outputs/viz/` 中查看 `episode_XX_overview.png` 和 `episode_XX_montage.png`。

---

### 步骤 2：运行单元测试（快速验证）

运行所有单元测试套件，确认基础功能正常。

**CMD:**
```cmd
python tests/run_all_tests.py
```

**Bash:**
```bash
python tests/run_all_tests.py
```

**验证结果：**
- 应看到 `Results: 8 passed, 0 failed out of 8 tests`
- 8 个测试套件包括：
  - `test_virtual_devices` — 虚拟硬件设备
  - `test_preprocessor` — 数据预处理
  - `test_dataset` — 数据集加载
  - `test_action_model` — 动作模型组件
  - `test_video_model` — 视频模型组件
  - `test_e2e` — 集成测试
  - `test_collect_ui` — 数据采集 UI 组件
  - `test_e2e_pipeline` — 完整流水线测试

---

### 步骤 3：训练动作预测模型

使用测试配置训练动作预测模型（3 个 epoch）。

**CMD:**
```cmd
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3
```

**Bash:**
```bash
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3
```

**验证结果：**
- 应看到 `Device: cuda`（或 `cpu`）
- 应看到 `Model params: 1.3M`
- 每个 epoch 显示 loss 值
- 最后应显示 `Training complete`
- 检查 checkpoint 文件：
  ```cmd
  dir outputs\test\checkpoints\
  ```
  应有 `action_best.pt` 和 `action_ckpt_epoch*.pt`

---

> **💡 可视化提示：** 训练前可再次打开 `python scripts/viz_mock_data.py` 确认训练数据的 action delta 分布合理。

---

### 步骤 4：训练视频预测模型

使用测试配置训练视频预测模型（3 个 epoch）。

**CMD:**
```cmd
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3
```

**Bash:**
```bash
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3
```

**验证结果：**
- 应看到 `Video model params: 5.4M`
- 每个 epoch 显示 loss 值
- 最后应显示 `Training complete`
- 检查 checkpoint：
  ```cmd
  dir outputs\test\checkpoints\
  ```
  应有 `video_best.pt`

---

### 步骤 5：评估模型

使用训练好的 checkpoint 评估两个模型。

**CMD:**
```cmd
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml
```

**Bash:**
```bash
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml
```

**验证结果：**

动作模型评估指标：
- `action_mse` — 动作均方误差（应为有限数值）
- `action_mae` — 动作平均绝对误差
- `endpoint_error` — 终点误差
- `mse_stage_dx`, `mse_stage_dy` — Stage XY 方向 MSE
- `mse_pip_dx`, `mse_pip_dy`, `mse_pip_dz` — Pipette XYZ 方向 MSE

视频模型评估指标：
- `pixel_mse` — 像素均方误差
- `pixel_mae` — 像素平均绝对误差
- `psnr` — 峰值信噪比
- `ssim` — 结构相似性
- `temporal_consistency` — 时间一致性

检查输出文件：
```cmd
type outputs\test\eval\eval_action.json
type outputs\test\eval\eval_video.json
```

---

### 步骤 5.5：可视化评估结果（推荐）

评估完成后，可视化对比预测值与真值。

**交互式检查：**
```cmd
python scripts/viz_mock_data.py --data_dir ./data/test_raw
```
逐帧查看原始数据，与 `outputs/test/eval/eval_action.json` 和 `eval_video.json` 中的指标对照。

**检查要点：**
- `action_mse` / `endpoint_error` 数值合理（不应为 NaN/Inf）
- 视频指标 `psnr` > 20dB（测试配置下可能较低，属正常）
- `temporal_consistency` 接近 1.0 表示时间连贯性好

---

### 步骤 6：运行推理

使用训练好的模型进行推理预测。

**CMD:**
```cmd
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"
```

**Bash:**
```bash
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"
```

**验证结果：**
- 应显示 `Predictor loaded on cpu`

**Demo 模式（无需 checkpoint）：**

```cmd
python inference/predict.py --demo --device cpu
```

**验证结果：**
- 应显示 `Actions shape: (16, 5)`
- 应显示 `Predicted frames shape: (4, 1, 384, 512)`

---

### 步骤 6.5：可视化推理结果（推荐）

推理完成后，用可视化工具对比输入帧与预测输出。

**交互式检查推理输入：**
```cmd
python scripts/viz_mock_data.py --data_dir ./data/test_raw
```
选择任意 episode，逐帧播放观察模型输入数据：
- 帧内容是否清晰
- 位置轨迹是否平滑
- Action delta 幅度是否合理

**检查要点：**
- 推理输出 `Actions shape: (16, 5)` — 5-DOF 动作序列
- 预测帧 `shape: (4, 1, 384, 512)` — 未来 4 帧视频
- 对比 `viz_mock_data.py` 中同 episode 的真值轨迹与推理输出

---

### 步骤 7：数据采集 UI（可选）

启动数据采集 GUI 界面。

**CMD:**
```cmd
python scripts/collect_ui.py
```

**Bash:**
```bash
python scripts/collect_ui.py
```

**验证结果：**
- 应弹出 tkinter 窗口
- 包含：摄像头预览、Stage 控制、Pipette 控制、PID 控制、录制面板

---

### 步骤 8：清理测试数据

测试完成后清理生成的临时文件。

**CMD:**
```cmd
rmdir /s /q data\test_raw
rmdir /s /q outputs\test
```

**Bash:**
```bash
rm -rf ./data/test_raw ./outputs/test
```

---

## 预期耗时

| 步骤 | 预期耗时 | 说明 |
|------|---------|------|
| 1. 生成数据 | ~5 秒 | 10 episodes, 200x160 分辨率 |
| 1.5. 可视化检查 | 手动 | `viz_mock_data.py` 交互式查看，可选 |
| 2. 单元测试 | ~60 秒 | 8 个测试套件, ~38 个测试 |
| 3. 训练动作模型 | ~15 秒 | 3 epochs, GPU |
| 4. 训练视频模型 | ~10 秒 | 3 epochs, GPU |
| 5. 评估模型 | ~90 秒 | 动作+视频评估 |
| 5.5. 可视化评估 | 手动 | 对照指标检查数据，可选 |
| 6. 推理 | ~30 秒 | 含模型加载 |
| 6.5. 可视化推理 | 手动 | 对比预测与真值，可选 |
| **总计** | **~3-4 分钟** | 自动步骤（不含可视化） |

> 注：CPU 模式下训练和推理耗时约为 GPU 的 3-5 倍。

---

## 故障排除

### 问题 1：`ModuleNotFoundError: No module named 'xxx'`

**解决方案：** 确保已激活 conda 环境并安装所有依赖：
```cmd
conda activate microdreamer
pip install -r requirements.txt
```

### 问题 2：`torch.load` 报错 `WeightsUnpickler error`

**原因：** PyTorch 2.6+ 默认 `weights_only=True`

**解决方案：** 已在代码中修复，所有 `torch.load` 调用已添加 `weights_only=False`

### 问题 3：CUDA 内存不足 (OOM)

**解决方案：** 使用 `config/test.yaml` 已配置小模型（128 hidden_dim, 2 layers），应可在任何 GPU 上运行。如仍 OOM，设置 `--device cpu`。

### 问题 4：评估时 `resolution` 不匹配

**原因：** 配置文件中 resolution 为 `[W, H]` 格式，模型内部使用 `(H, W)` 格式

**解决方案：** 已在代码中修复，evaluate.py 和 predict.py 已正确转换

### 问题 5：`num_tiles` 不匹配

**原因：** 测试配置使用 `num_tiles: 1`（小分辨率），但某些脚本未从配置读取

**解决方案：** 已在代码中修复，所有模型创建点已正确传入 `num_tiles` 配置

---

## 测试配置说明

使用 `config/test.yaml` 而非 `config/default.yaml`，参数对比：

| 参数 | default.yaml | test.yaml | 说明 |
|------|-------------|-----------|------|
| camera resolution | 1600x1200 | 200x160 | 小分辨率加速测试 |
| hidden_dim | 512 | 128 | 小模型加速 |
| num_layers | 6 | 2 | 更少层 |
| num_heads | 8 | 4 | 更少头 |
| action_horizon | 16 | 4 | 更短预测 |
| num_frames | 16 | 8 | 更少帧 |
| max_epochs | 100 | 3 | 快速验证 |
| batch_size | 2 | 2 | 保持不变 |
| fp16 | true | false | CPU 兼容 |
| deepspeed | true | false | 单 GPU 测试 |

---

## 所有脚本 `--help` 命令速查

```cmd
python scripts/generate_test_data.py --help
python scripts/collect_data.py --help
python scripts/collect_ui.py --help
python scripts/train_action.py --help
python scripts/train_video.py --help
python scripts/evaluate.py --help
python scripts/calibrate.py --help
python inference/predict.py --help
python scripts/viz_mock_data.py --help
python scripts/visualize_mock_data.py --help
```
