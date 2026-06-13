# MicroDreamer 数据采集指南

[English](./DATA_COLLECTION_GUIDE_EN.md) | 中文

本文档详细描述了 MicroDreamer 项目的数据采集流程、标注方法和注意事项。

---

## 目录

1. [概述](#1-概述)
2. [硬件环境](#2-硬件环境)
3. [采集前准备](#3-采集前准备)
4. [数据采集流程](#4-数据采集流程)
5. [标注规范](#5-标注规范)
6. [数据格式说明](#6-数据格式说明)
7. [质量控制](#7-质量控制)
8. [常见问题与注意事项](#8-常见问题与注意事项)

---

## 1. 概述

MicroDreamer 需要两类数据来训练：
- **视频帧序列**: 显微镜下的高分辨率图像 (1600×1200, 灰度)
- **5-DOF 动作数据**: stage_x, stage_y, pipette_x, pipette_y, pipette_z

数据以 **episode（片段）** 为单位采集，每个 episode 包含：
- 一段连续的帧序列（通常 100-500 帧，对应 3-17 秒 @30fps）
- 每帧对应的 5-DOF 位置数据
- 语言任务描述（如 "move to cell and pick up"）

### 数据量建议

| 训练阶段 | Episode 数量 | 每 Episode 帧数 | 总帧数 |
|---------|------------|---------------|-------|
| 初步测试 | 50-100 | 100-200 | 5K-20K |
| 正式训练 | 500-2000 | 200-500 | 100K-1M |
| 大规模训练 | 5000+ | 300-500 | 1.5M+ |

---

## 2. 硬件环境

### 2.1 显微镜系统

- **显微镜型号**: Nikon Ti2E（或兼容型号）
- **相机**: Basler 相机，1600×1200 分辨率，Mono8 灰度模式
- **帧率**: 30 fps
- **像素分辨率**: 0.6 μm/pixel（可通过标定调整）
- **物镜**: 建议 10× 或 20× 物镜

### 2.2 运动控制

- **XY Stage**: Nikon Ti2E 电动载物台
  - 行程范围: ~110mm × 75mm
  - 重复精度: < 1 μm
  - 控制频率: 100 Hz
- **Pipette (手臂)**: MMS Motor 控制
  - 3-DOF: X, Y, Z
  - Z 轴用于控制针尖下降/上升
  - 控制频率: 100 Hz

### 2.3 计算机配置

- **操作系统**: Windows 10/11
- **GPU**: NVIDIA GPU，显存 ≥ 16GB（推荐 RTX 4090 或 5090）
- **内存**: ≥ 32GB
- **存储**: SSD，≥ 500GB（用于存储原始数据）

---

## 3. 采集前准备

### 3.1 环境配置

```bash
# 激活 conda 环境
conda activate microdreamer

# 确认依赖已安装
pip install -r requirements.txt

# 运行测试确认代码正常
python tests/run_all_tests.py
```

### 3.2 硬件连接检查

1. **相机连接**:
   - 确认 Basler 相机已连接并被系统识别
   - 运行 `python scripts/calibrate.py --mode pixel` 验证相机工作正常
   - 检查图像是否清晰，亮度是否合适

2. **Stage 连接**:
   - 确认 Stage 串口连接正常
   - 运行 `python scripts/calibrate.py --mode focus` 验证 Stage 移动正常
   - 检查 Stage 是否可以自由移动到全行程范围

3. **Pipette 连接**:
   - 确认 MMS Motor HTTP 服务运行在 localhost:5000
   - 验证 Pipette X/Y/Z 轴均可正常控制
   - 确认 Z 轴安全限位设置正确

### 3.3 样品准备

1. **细胞/样品**:
   - 确保样品已正确放置在培养皿中
   - 检查培养液状态，确保细胞活性良好
   - 记录样品类型和状态

2. **Pipette 准备**:
   - 安装合适直径的 Pipette（根据细胞大小选择）
   - 确认 Pipette 尖端清洁无堵塞
   - 调整 Pipette 初始位置（Z 轴高于样品平面）

### 3.4 参数标定

```bash
# 像素到微米的转换标定
python scripts/calibrate.py --mode pixel --pixel_size 0.6

# 焦平面标定
python scripts/calibrate.py --mode focus

# 完整标定
python scripts/calibrate.py --mode all
```

标定结果会保存到 `calibration/` 目录，后续采集时自动加载。

---

## 4. 数据采集流程

### 4.1 使用 UI 采集（推荐）

```bash
# 启动数据采集 UI
python scripts/collect_ui.py
```

UI 界面包含：
- **相机预览窗口**: 实时显示显微镜画面
- **Stage 控制面板**: 手动移动 + PID 自动定位
- **Pipette 控制面板**: 手动移动 + Z 轴控制
- **数据采集控制**: 开始/停止录制、任务描述输入

### 4.2 使用命令行采集

```bash
# 虚拟模式（测试用）
python scripts/collect_data.py --mode virtual --num_episodes 10

# 真实模式
python scripts/collect_data.py --mode real --num_episodes 50 --task_description "pick up cell"
```

### 4.3 采集步骤详解

#### 步骤 1: 启动系统

1. 打开显微镜电源，等待系统稳定（约 5 分钟）
2. 启动 MMS Motor 服务
3. 运行 `python scripts/collect_ui.py` 启动采集 UI
4. 在 UI 中确认相机、Stage、Pipette 均已连接（状态栏显示绿色）

#### 步骤 2: 调整视野

1. 使用 Stage 控制面板手动移动，找到目标区域
2. 调整焦平面，确保细胞清晰可见
3. 调整相机曝光和增益，确保图像亮度合适
4. 可选：使用 PID 自动定位功能，点击画面中的目标位置，Stage 会自动移动到该位置

#### 步骤 3: 设置任务描述

1. 在 UI 的"任务描述"输入框中输入当前任务的自然语言描述
2. 示例：
   - "move to cell and pick up"
   - "transfer cell to target location"
   - "inject cell with substance"
   - "sort cells by size"
3. 确保描述准确反映当前操作

#### 步骤 4: 开始录制

1. 点击"开始录制"按钮
2. UI 状态栏会显示录制状态和当前帧数
3. 开始执行操作：
   - 使用 Stage 控制移动视野
   - 使用 Pipette 控制进行操作
   - 所有移动和操作会被自动记录

#### 步骤 5: 执行操作

操作过程中注意：
- **保持平稳**: 避免突然的大幅度移动，保持操作平稳
- **完整动作**: 确保每个动作从开始到结束都被完整记录
- **适当速度**: 移动速度不宜过快（建议 Stage < 100 μm/s）
- **Z 轴安全**: 下降 Pipette 时注意不要撞到样品或培养皿底部

#### 步骤 6: 停止录制

1. 操作完成后，点击"停止录制"按钮
2. 系统会自动保存数据到 `data/raw/` 目录
3. 每个 episode 保存为一个子目录，包含：
   - `data.npz`: 帧序列和位置数据
   - `metadata.json`: 元数据（任务描述、时间戳等）

#### 步骤 7: 检查数据质量

录制完成后，检查：
1. 帧序列是否完整，无丢帧
2. 位置数据是否连续，无跳变
3. 任务描述是否准确
4. 图像质量是否清晰

---

## 5. 标注规范

### 5.1 语言标注

#### 任务描述模板

MicroDreamer 使用自然语言描述任务。标注时应遵循以下规范：

**基本格式**:
```
[动作动词] [目标对象] [附加条件/方式]
```

**常用动词**:
- `move to` - 移动到
- `pick up` / `capture` - 抓取/捕获
- `inject` - 注射
- `transfer` - 转移
- `sort` - 分选
- `position` - 定位
- `align` - 对齐
- `approach` - 接近

**示例标注**:

| 操作类型 | 任务描述示例 |
|---------|------------|
| 单步移动 | "move stage to cell cluster at center" |
| 抓取操作 | "approach cell with pipette and pick up" |
| 注射操作 | "position pipette above cell, descend, and inject" |
| 转移操作 | "pick up cell from source and transfer to target location" |
| 分选操作 | "sort cells by size, move large cells to left" |
| 对齐操作 | "align pipette tip with cell center" |

#### 标注注意事项

1. **准确性**: 描述必须准确反映实际操作，不要夸大或简化
2. **完整性**: 包含所有关键步骤，不要遗漏重要动作
3. **一致性**: 同类操作使用相同的描述格式
4. **简洁性**: 在准确的前提下尽量简洁，避免冗余描述
5. **时序性**: 如果操作有明确顺序，描述中应体现（如 "first...then..."）

### 5.2 动作标注

动作数据由系统自动记录，无需手动标注。但需要确保：

1. **坐标系一致**: 所有数据使用相同的坐标系
2. **单位统一**: Stage 位置单位为 μm，Pipette 位置单位为 μm
3. **时间对齐**: 帧和位置数据时间戳对齐（< 50ms 误差）

### 5.3 数据分割

每个 episode 应该是一个完整的操作单元：

- **开始状态**: 操作开始前的稳定状态
- **操作过程**: 完整的操作动作序列
- **结束状态**: 操作完成后的稳定状态
- **建议帧数**: 100-500 帧（3-17 秒 @30fps）

---

## 6. 数据格式说明

### 6.1 目录结构

```
data/
├── raw/                          # 原始采集数据
│   ├── episode_20260613_143022/  # 每个 episode 一个目录
│   │   ├── data.npz             # 帧序列 + 位置数据
│   │   └── metadata.json        # 元数据
│   ├── episode_20260613_143156/
│   │   ├── data.npz
│   │   └── metadata.json
│   └── ...
├── processed/                    # 预处理后的数据
│   ├── actions_normalized.npz   # 归一化的动作数据
│   └── normalizer_stats.npz    # 归一化参数
└── dummy_train/                  # 测试用虚拟数据
```

### 6.2 data.npz 格式

```python
{
    'frames': np.ndarray,        # shape (N, 1200, 1600), dtype uint8, 灰度帧
    'stage_positions': np.ndarray,  # shape (N, 2), dtype float32, [x, y] μm
    'pipette_positions': np.ndarray,  # shape (N, 3), dtype float32, [x, y, z] μm
}
```

### 6.3 metadata.json 格式

```json
{
    "episode_id": "episode_20260613_143022",
    "timestamp": "2026-06-13T14:30:22.123456",
    "task_description": "move to cell and pick up",
    "num_frames": 300,
    "camera_fps": 30.0,
    "camera_resolution": [1600, 1200],
    "pixel_size_um": 0.6,
    "stage_range_x": [0.0, 110000.0],
    "stage_range_y": [0.0, 75000.0],
    "pipette_range_z": [0.0, 200.0],
    "notes": ""
}
```

---

## 7. 质量控制

### 7.1 采集时检查

| 检查项 | 标准 | 处理方式 |
|-------|------|---------|
| 帧率稳定性 | 实际帧率 ≥ 28 fps | 检查相机设置和系统负载 |
| 图像清晰度 | 细胞边缘清晰可辨 | 调整焦平面和光源 |
| 位置数据连续性 | 无跳变，变化平滑 | 检查 Stage/Pipette 连接 |
| 时间对齐 | 帧-位置时间差 < 50ms | 检查同步器设置 |
| 数据完整性 | 每帧都有对应位置数据 | 检查同步器缓冲区 |

### 7.2 采集后验证

```bash
# 验证数据完整性
python scripts/validate_data.py --data_dir data/raw

# 检查数据统计
python scripts/data_stats.py --data_dir data/raw
```

### 7.3 常见数据问题

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 丢帧 | 相机帧率不足或系统负载高 | 降低分辨率或关闭其他程序 |
| 位置跳变 | Stage/Pipette 通信中断 | 检查连接线和驱动 |
| 图像模糊 | 焦平面偏移或振动 | 重新对焦，消除振动源 |
| 数据不同步 | 同步器配置错误 | 调整同步器参数 |
| 帧数为 0 | 相机未正确初始化 | 重启相机驱动 |

---

## 8. 常见问题与注意事项

### 8.1 安全注意事项

1. **Z 轴安全**:
   - 下降 Pipette 前确认 Z 轴位置安全
   - 设置 Z 轴最低限位，防止撞到培养皿
   - 使用 UI 时注意 Z 轴位置显示

2. **样品保护**:
   - 避免 Pipette 碰撞样品
   - 操作前确认 Pipette 尖端位置
   - 保持适当的操作速度

3. **设备保护**:
   - 不要超出 Stage 行程范围
   - 避免 Pipette 碰撞培养皿边缘
   - 定期检查设备状态

### 8.2 数据采集技巧

1. **多样化数据**:
   - 采集不同位置、不同细胞的操作数据
   - 包含成功和失败的操作（用于训练鲁棒性）
   - 变化操作速度和路径

2. **任务描述一致性**:
   - 建立标准的任务描述词表
   - 同类操作使用相同的描述格式
   - 记录特殊情况的说明

3. **批次管理**:
   - 按日期和样品分批采集
   - 记录每批数据的样品信息
   - 定期备份原始数据

### 8.3 性能优化

1. **存储优化**:
   - 使用 SSD 存储原始数据
   - 定期清理不需要的中间文件
   - 压缩存储长期保存的数据

2. **采集效率**:
   - 使用 UI 的 PID 自动定位功能快速定位目标
   - 预设常用的操作参数
   - 批量采集相似操作

### 8.4 故障排除

**问题**: 相机无法连接
```
解决方案:
1. 检查相机 USB/网线连接
2. 确认 Basler pylon 驱动已安装
3. 运行 python -c "from hardware.camera.basler_camera import BaslerCamera; cam = BaslerCamera(); cam.open(); print('OK')"
4. 检查相机是否被其他程序占用
```

**问题**: Stage 移动不响应
```
解决方案:
1. 检查 Stage 串口连接
2. 确认 Stage 电源已开启
3. 检查 StageCPP.dll 是否正确加载
4. 运行 python scripts/calibrate.py --mode focus 测试
```

**问题**: Pipette 控制失败
```
解决方案:
1. 确认 MMS Motor 服务运行在 localhost:5000
2. 检查 HTTP 连接: curl http://localhost:5000/status
3. 确认 Pipette 电机电源已开启
4. 检查安全限位设置
```

**问题**: 数据采集丢帧
```
解决方案:
1. 降低相机帧率（从 30fps 降到 15fps）
2. 关闭其他占用 CPU/GPU 的程序
3. 检查硬盘写入速度
4. 减少同时运行的采集设备数量
```

---

## 附录 A: 快速参考卡

### 启动命令

```bash
# 启动采集 UI
python scripts/collect_ui.py

# 命令行采集（虚拟模式）
python scripts/collect_data.py --mode virtual --num_episodes 10

# 命令行采集（真实模式）
python scripts/collect_data.py --mode real --num_episodes 50

# 标定
python scripts/calibrate.py --mode all

# 数据验证
python scripts/validate_data.py --data_dir data/raw
```

### 常用参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| camera_fps | 30 | 相机帧率 |
| camera_resolution | [1600, 1200] | 相机分辨率 |
| pixel_size_um | 0.6 | 像素尺寸 (μm/pixel) |
| sync_tolerance_ms | 50 | 同步容差 (ms) |
| stage_max_speed | 100 | Stage 最大速度 (μm/s) |
| pipette_z_safe | 50 | Pipette Z 轴安全高度 (μm) |

### 任务描述模板

```
# 单步操作
"move stage to [目标位置]"
"position pipette above [目标]"
"descend pipette to [深度]"

# 复合操作
"approach [目标] with pipette and pick up"
"transfer [目标] from [来源] to [目标位置]"
"inject [目标] with [物质]"

# 条件操作
"sort [目标] by [属性]"
"align [目标A] with [目标B]"
"monitor [目标] for [时间]"
```

---

## 附录 B: 数据采集检查清单

采集前：
- [ ] 显微镜电源已开启，系统已稳定
- [ ] 相机连接正常，图像清晰
- [ ] Stage 连接正常，可自由移动
- [ ] Pipette 连接正常，MMS 服务运行
- [ ] 标定文件已加载
- [ ] 样品已正确放置
- [ ] 任务描述已准备

采集中：
- [ ] 帧率稳定 ≥ 28 fps
- [ ] 位置数据连续无跳变
- [ ] 图像质量清晰
- [ ] 操作平稳，无突然移动
- [ ] 动作完整，从开始到结束

采集后：
- [ ] 数据已保存到正确目录
- [ ] metadata.json 信息完整
- [ ] 帧数和位置数据数量匹配
- [ ] 任务描述准确
- [ ] 数据质量检查通过

---

*最后更新: 2026-06-13*
