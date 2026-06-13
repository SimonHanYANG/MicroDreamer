# MicroDreamer Data Collection Guide

English | [中文](./DATA_COLLECTION_GUIDE.md)

This document describes the data collection workflow, annotation standards, and best practices for the MicroDreamer project.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware Setup](#2-hardware-setup)
3. [Pre-Collection Setup](#3-pre-collection-setup)
4. [Collection Workflow](#4-collection-workflow)
5. [Annotation Standards](#5-annotation-standards)
6. [Data Format](#6-data-format)
7. [Quality Control](#7-quality-control)
8. [Troubleshooting & Best Practices](#8-troubleshooting--best-practices)

---

## 1. Overview

MicroDreamer requires two types of data for training:
- **Video frame sequences**: High-resolution microscopy images (1600×1200, grayscale)
- **5-DOF action data**: stage_x, stage_y, pipette_x, pipette_y, pipette_z

Data is collected in **episodes**. Each episode contains:
- A continuous frame sequence (typically 100-500 frames, 3-17 seconds @30fps)
- Corresponding 5-DOF position data per frame
- Natural language task description (e.g., "move to cell and pick up")

### Data Volume Recommendations

| Training Phase | Episodes | Frames/Episode | Total Frames |
|---------------|----------|----------------|--------------|
| Initial Testing | 50-100 | 100-200 | 5K-20K |
| Standard Training | 500-2000 | 200-500 | 100K-1M |
| Large-Scale Training | 5000+ | 300-500 | 1.5M+ |

---

## 2. Hardware Setup

### 2.1 Microscope System

- **Microscope**: Nikon Ti2E (or compatible)
- **Camera**: Basler camera, 1600×1200 resolution, Mono8 grayscale
- **Frame Rate**: 30 fps
- **Pixel Resolution**: 0.6 μm/pixel (adjustable via calibration)
- **Objective**: 10× or 20× recommended

### 2.2 Motion Control

- **XY Stage**: Nikon Ti2E motorized stage
  - Travel range: ~110mm × 75mm
  - Repeatability: < 1 μm
  - Control frequency: 100 Hz
- **Pipette (Arm)**: MMS Motor control
  - 3-DOF: X, Y, Z
  - Z-axis for pipette descent/ascent
  - Control frequency: 100 Hz

### 2.3 Computer Requirements

- **OS**: Windows 10/11
- **GPU**: NVIDIA GPU with ≥ 16GB VRAM (RTX 4090 or 5090 recommended)
- **RAM**: ≥ 32GB
- **Storage**: SSD with ≥ 500GB for raw data

---

## 3. Pre-Collection Setup

### 3.1 Environment Configuration

```bash
# Activate conda environment
conda activate microdreamer

# Verify dependencies
pip install -r requirements.txt

# Run tests to confirm code works
python tests/run_all_tests.py
```

### 3.2 Hardware Connection Check

1. **Camera**:
   - Confirm Basler camera is connected and recognized
   - Run `python scripts/calibrate.py --mode pixel` to verify camera works
   - Check image clarity and brightness

2. **Stage**:
   - Confirm Stage serial connection is active
   - Run `python scripts/calibrate.py --mode focus` to verify Stage movement
   - Check Stage can move freely across full range

3. **Pipette**:
   - Confirm MMS Motor HTTP service is running at localhost:5000
   - Verify Pipette X/Y/Z axes can be controlled
   - Confirm Z-axis safety limits are set

### 3.3 Sample Preparation

1. **Cells/Samples**:
   - Ensure samples are properly placed in culture dish
   - Check culture medium status, ensure cell viability
   - Record sample type and condition

2. **Pipette Preparation**:
   - Install appropriate diameter pipette (based on cell size)
   - Confirm pipette tip is clean and unobstructed
   - Adjust initial pipette position (Z-axis above sample plane)

### 3.4 Calibration

```bash
# Pixel-to-micrometer calibration
python scripts/calibrate.py --mode pixel --pixel_size 0.6

# Focus plane calibration
python scripts/calibrate.py --mode focus

# Full calibration
python scripts/calibrate.py --mode all
```

Calibration results are saved to `calibration/` directory and automatically loaded during collection.

---

## 4. Collection Workflow

### 4.1 Using the UI (Recommended)

```bash
# Launch data collection UI
python scripts/collect_ui.py
```

The UI includes:
- **Camera preview window**: Real-time microscope view
- **Stage control panel**: Manual movement + PID auto-positioning
- **Pipette control panel**: Manual movement + Z-axis control
- **Data collection controls**: Start/stop recording, task description input

### 4.2 Using Command Line

```bash
# Virtual mode (for testing)
python scripts/collect_data.py --mode virtual --num_episodes 10

# Real mode
python scripts/collect_data.py --mode real --num_episodes 50 --task_description "pick up cell"
```

### 4.3 Detailed Collection Steps

#### Step 1: System Startup

1. Turn on microscope power, wait for system to stabilize (~5 minutes)
2. Start MMS Motor service
3. Run `python scripts/collect_ui.py` to launch collection UI
4. In UI, confirm camera, stage, and pipette are connected (status bar shows green)

#### Step 2: Adjust Field of View

1. Use Stage control panel to manually move and find target area
2. Adjust focal plane to ensure cells are clearly visible
3. Adjust camera exposure and gain for proper brightness
4. Optional: Use PID auto-positioning - click target position in camera view, stage will auto-move

#### Step 3: Set Task Description

1. Enter natural language task description in the UI's "Task Description" field
2. Examples:
   - "move to cell and pick up"
   - "transfer cell to target location"
   - "inject cell with substance"
   - "sort cells by size"
3. Ensure description accurately reflects the current operation

#### Step 4: Start Recording

1. Click "Start Recording" button
2. UI status bar shows recording status and current frame count
3. Begin operation:
   - Use Stage control to move field of view
   - Use Pipette control for manipulation
   - All movements and operations are automatically recorded

#### Step 5: Execute Operation

During operation:
- **Keep steady**: Avoid sudden large movements
- **Complete actions**: Ensure each action is fully recorded from start to end
- **Appropriate speed**: Movement speed should not be too fast (Stage < 100 μm/s recommended)
- **Z-axis safety**: When lowering pipette, be careful not to hit sample or dish bottom

#### Step 6: Stop Recording

1. After operation is complete, click "Stop Recording" button
2. System automatically saves data to `data/raw/` directory
3. Each episode is saved as a subdirectory containing:
   - `data.npz`: Frame sequence and position data
   - `metadata.json`: Metadata (task description, timestamps, etc.)

#### Step 7: Check Data Quality

After recording, verify:
1. Frame sequence is complete, no dropped frames
2. Position data is continuous, no jumps
3. Task description is accurate
4. Image quality is clear

---

## 5. Annotation Standards

### 5.1 Language Annotation

#### Task Description Templates

MicroDreamer uses natural language to describe tasks. Follow these standards:

**Basic Format**:
```
[action verb] [target object] [additional conditions/method]
```

**Common Verbs**:
- `move to` - Move to
- `pick up` / `capture` - Capture/grab
- `inject` - Inject
- `transfer` - Transfer
- `sort` - Sort
- `position` - Position
- `align` - Align
- `approach` - Approach

**Example Annotations**:

| Operation Type | Task Description Example |
|---------------|-------------------------|
| Simple Move | "move stage to cell cluster at center" |
| Capture | "approach cell with pipette and pick up" |
| Injection | "position pipette above cell, descend, and inject" |
| Transfer | "pick up cell from source and transfer to target location" |
| Sorting | "sort cells by size, move large cells to left" |
| Alignment | "align pipette tip with cell center" |

#### Annotation Guidelines

1. **Accuracy**: Description must accurately reflect actual operation
2. **Completeness**: Include all key steps, don't omit important actions
3. **Consistency**: Use same description format for same type of operations
4. **Conciseness**: Be concise while maintaining accuracy
5. **Temporal Order**: If operations have clear sequence, reflect in description

### 5.2 Action Annotation

Action data is automatically recorded by the system. Ensure:

1. **Coordinate System**: All data uses the same coordinate system
2. **Unit Consistency**: Stage positions in μm, Pipette positions in μm
3. **Time Alignment**: Frame and position data timestamps aligned (< 50ms error)

### 5.3 Episode Segmentation

Each episode should be a complete operation unit:

- **Start State**: Stable state before operation begins
- **Operation**: Complete action sequence
- **End State**: Stable state after operation completes
- **Recommended Frames**: 100-500 frames (3-17 seconds @30fps)

---

## 6. Data Format

### 6.1 Directory Structure

```
data/
├── raw/                          # Raw collected data
│   ├── episode_20260613_143022/  # One directory per episode
│   │   ├── data.npz             # Frame sequence + position data
│   │   └── metadata.json        # Metadata
│   ├── episode_20260613_143156/
│   │   ├── data.npz
│   │   └── metadata.json
│   └── ...
├── processed/                    # Preprocessed data
│   ├── actions_normalized.npz   # Normalized action data
│   └── normalizer_stats.npz    # Normalization parameters
└── dummy_train/                  # Virtual test data
```

### 6.2 data.npz Format

```python
{
    'frames': np.ndarray,        # shape (N, 1200, 1600), dtype uint8, grayscale frames
    'stage_positions': np.ndarray,  # shape (N, 2), dtype float32, [x, y] in μm
    'pipette_positions': np.ndarray,  # shape (N, 3), dtype float32, [x, y, z] in μm
}
```

### 6.3 metadata.json Format

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

## 7. Quality Control

### 7.1 During Collection

| Check Item | Standard | Action |
|-----------|----------|--------|
| Frame Rate Stability | Actual rate ≥ 28 fps | Check camera settings and system load |
| Image Clarity | Cell edges clearly visible | Adjust focal plane and lighting |
| Position Data Continuity | No jumps, smooth changes | Check Stage/Pipette connections |
| Time Alignment | Frame-position time diff < 50ms | Check synchronizer settings |
| Data Completeness | Every frame has position data | Check synchronizer buffer |

### 7.2 Post-Collection Validation

```bash
# Validate data integrity
python scripts/validate_data.py --data_dir data/raw

# Check data statistics
python scripts/data_stats.py --data_dir data/raw
```

### 7.3 Common Data Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Dropped frames | Camera frame rate too low or high system load | Reduce resolution or close other programs |
| Position jumps | Stage/Pipette communication interrupt | Check connection cables and drivers |
| Blurry images | Focal plane drift or vibration | Re-focus, eliminate vibration source |
| Data out of sync | Synchronizer config error | Adjust synchronizer parameters |
| Zero frames | Camera not properly initialized | Restart camera driver |

---

## 8. Troubleshooting & Best Practices

### 8.1 Safety Notes

1. **Z-Axis Safety**:
   - Confirm Z-axis position is safe before lowering pipette
   - Set Z-axis minimum limit to prevent hitting culture dish
   - Monitor Z-axis position display in UI

2. **Sample Protection**:
   - Avoid pipette collision with samples
   - Confirm pipette tip position before operation
   - Maintain appropriate operation speed

3. **Equipment Protection**:
   - Do not exceed Stage travel range
   - Avoid pipette collision with dish edges
   - Regularly check equipment status

### 8.2 Data Collection Tips

1. **Data Diversity**:
   - Collect data from different positions and different cells
   - Include both successful and failed operations (for robustness training)
   - Vary operation speeds and paths

2. **Task Description Consistency**:
   - Establish standard task description vocabulary
   - Use same description format for same type of operations
   - Record notes for special cases

3. **Batch Management**:
   - Collect in batches by date and sample
   - Record sample information for each batch
   - Regularly backup raw data

### 8.3 Performance Optimization

1. **Storage Optimization**:
   - Use SSD for raw data storage
   - Regularly clean unnecessary intermediate files
   - Compress data for long-term storage

2. **Collection Efficiency**:
   - Use UI's PID auto-positioning for quick target acquisition
   - Preset commonly used operation parameters
   - Batch collect similar operations

### 8.4 Troubleshooting

**Problem**: Camera cannot connect
```
Solution:
1. Check camera USB/network cable connection
2. Confirm Basler pylon driver is installed
3. Run python -c "from hardware.camera.basler_camera import BaslerCamera; cam = BaslerCamera(); cam.open(); print('OK')"
4. Check if camera is occupied by another program
```

**Problem**: Stage not responding
```
Solution:
1. Check Stage serial connection
2. Confirm Stage power is on
3. Check if StageCPP.dll loads correctly
4. Run python scripts/calibrate.py --mode focus to test
```

**Problem**: Pipette control failure
```
Solution:
1. Confirm MMS Motor service is running at localhost:5000
2. Check HTTP connection: curl http://localhost:5000/status
3. Confirm Pipette motor power is on
4. Check safety limit settings
```

**Problem**: Data collection dropping frames
```
Solution:
1. Reduce camera frame rate (from 30fps to 15fps)
2. Close other CPU/GPU intensive programs
3. Check disk write speed
4. Reduce number of simultaneously running collection devices
```

---

## Appendix A: Quick Reference Card

### Launch Commands

```bash
# Launch collection UI
python scripts/collect_ui.py

# Command line collection (virtual mode)
python scripts/collect_data.py --mode virtual --num_episodes 10

# Command line collection (real mode)
python scripts/collect_data.py --mode real --num_episodes 50

# Calibration
python scripts/calibrate.py --mode all

# Data validation
python scripts/validate_data.py --data_dir data/raw
```

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| camera_fps | 30 | Camera frame rate |
| camera_resolution | [1600, 1200] | Camera resolution |
| pixel_size_um | 0.6 | Pixel size (μm/pixel) |
| sync_tolerance_ms | 50 | Sync tolerance (ms) |
| stage_max_speed | 100 | Stage max speed (μm/s) |
| pipette_z_safe | 50 | Pipette Z-axis safe height (μm) |

### Task Description Templates

```
# Single-step operations
"move stage to [target position]"
"position pipette above [target]"
"descend pipette to [depth]"

# Compound operations
"approach [target] with pipette and pick up"
"transfer [target] from [source] to [destination]"
"inject [target] with [substance]"

# Conditional operations
"sort [target] by [property]"
"align [targetA] with [targetB]"
"monitor [target] for [duration]"
```

---

## Appendix B: Data Collection Checklist

Before Collection:
- [ ] Microscope power on, system stabilized
- [ ] Camera connected, image clear
- [ ] Stage connected, moves freely
- [ ] Pipette connected, MMS service running
- [ ] Calibration file loaded
- [ ] Sample properly placed
- [ ] Task description prepared

During Collection:
- [ ] Frame rate stable ≥ 28 fps
- [ ] Position data continuous, no jumps
- [ ] Image quality clear
- [ ] Operations steady, no sudden movements
- [ ] Actions complete from start to end

After Collection:
- [ ] Data saved to correct directory
- [ ] metadata.json information complete
- [ ] Frame count matches position data count
- [ ] Task description accurate
- [ ] Data quality check passed

---

*Last updated: 2026-06-13*
