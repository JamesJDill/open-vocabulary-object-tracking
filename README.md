# Grounded Multi-Object Tracking with GroundingDINO

A lightweight open-vocabulary multi-object tracker built on top of GroundingDINO detections. This project takes phrase-conditioned detections from GroundingDINO and links them over time using a Kalman-filter-based tracker with adaptive label-aware confidence thresholding, class-wise NMS, and a combined geometric-plus-appearance assignment cost to enable SORT-based tracking in the Open-Vocabulary scenarios. The result is a tracker that can follow both standard tracking prompts and unconventional prompts while preserving per-track identities across video frames.

## Examples
<table>
  <tr>
    <td align="center" width="50%">
      <b> ["person"] </b><br/>
      <img src="assets/demo2_person.gif" alt="Example result 1" width="100%"/><br/>
      <sub><code>assets/demo2_person.gif</code></sub>
    </td>
    <td align="center" width="50%">
      <b> ["shirt", "pants", "head", "backpack"] </b><br/>
      <img src="assets/demo2_body.gif" alt="Example result 2" width="100%"/><br/>
      <sub><code>assets/demo2_body.gif</code></sub>
    </td>
  </tr>
</table>

<br/>

<table>
  <tr>
    <td align="center" width="50%">
      <b> ["person"] </b><br/>
      <img src="assets/demo1_person.gif" alt="Example result 1" width="100%"/><br/>
      <sub><code>assets/demo1_person.gif</code></sub>
    </td>
    <td align="center" width="50%">
      <b> ["shirt", "pants", "head"] </b><br/>
      <img src="assets/demo1_body.gif" alt="Example result 2" width="100%"/><br/>
      <sub><code>assets/demo1_body.gif</code></sub>
    </td>
  </tr>
</table>

## Abstract

This repository implements an open-vocabulary tracking pipeline that decouples detection from association. GroundingDINO provides phrase-conditioned detections, while the core tracker abstraction performs temporal association using (1) a Kalman filter for motion prediction, (2) ByteTrack-style track associations, (3) adaptive per-label score thresholds computed from current-frame detections, (4) class-wise non-maximum suppression, and (5) a weighted assignment cost that blends geometry-based distance cost with score-vector appearance similarity. Each track also maintains an exponentially smoothed label-score appearance vector over time, enabling more stable matching when detector scores fluctuate frame to frame.

## Features

- Open-vocabulary detection backend via GroundingDINO
- Phrase-conditioned tracking from arbitrary label lists
- Per-label adaptive score thresholds
- Same-label-only NMS before tracking
- Kalman-filter motion modeling
- Two-pass association for high-confidence then recovery matching
- Combined assignment cost: geometric distance cost + appearance / score-vector cost
- Per-track appearance vectors updated with EMA smoothing
- Demo notebook for end-to-end video tracking in `demo/gdino_demo.ipynb`

## Repository Structure

```text
open-vocabulary-object-tracking/
├── demo/
│   ├── demo1.mp4
│   ├── demo2.mp4
│   └── gdino_demo.ipynb
├── external/
│   └── GroundingDINO/   # git submodule
├── tracker/
│   ├── assignment.py
│   ├── kalman_filter.py
│   ├── track.py
│   ├── tracker.py
│   └── utils.py
└── README.md
```

## Environment Set-up

### 1. Install CUDA Toolkit
Install the cuda-toolkit, if necessary refer to the [Official Documentation](https://docs.nvidia.com/cuda/) for your system. I recommend installing 12.1 or 12.8 which have known compatibility with GroundingDINO.

After installation you can test it like this:
```bash
which nvcc
nvcc --version
```

You should see something like this
```bash
/usr/local/cuda/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Mon_Mar_02_09:52:23_PM_PST_2026
Cuda compilation tools, release 13.2, V13.2.51
Build cuda_13.2.r13.2/compiler.37434383_0
```

Run this so the environment variable will be set under current shell.
```
export CUDA_HOME=/usr/local/cuda
```

Check your `$CUDA_HOME` variable
```bash
echo $CUDA_HOME
```
If it print nothing, then it means you haven't set up the path or you haven't installed the cuda-toolkit.


The version of cuda should be aligned with your CUDA runtime for there might exists multiple cuda at the same time. To check your version refer to `which nvcc`.

Run this so the environment variable will be set under current shell.
```bash
export CUDA_HOME=/path/to/cuda
```

If you want to set the CUDA_HOME permanently, store it using:
```bash
echo 'export CUDA_HOME=/path/to/cuda' >> ~/.bashrc
source ~/.bashrc
```

### 2. Repo + Environment Set-up

```bash
git clone --recurse-submodules https://github.com/JamesJDill/GroundedTrack.git
cd GroundedTrack

conda create -n ovtrack python=3.10 -y
conda activate ovtrack

# Install PyTorch compatible with your cuda-toolkit version (https://pytorch.org/get-started/locally/)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
```

### 3. Install GroundingDINO package
Full install instructions can be found at the [Official GroundingDINO Github Page](https://github.com/IDEA-Research/GroundingDINO)
```bash
pip install --no-build-isolation -e ./external/GroundingDINO

mkdir -p external/GroundingDINO/weights
cd external/GroundingDINO/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../..
```

## Demo

This repository includes a demo notebook for running open-vocabulary tracking end-to-end:
- `demo/gdino_demo.ipynb`
  
It also comes with two example demo videos:
- `demo/demo1.mp4`
- `demo/demo2.mp4`

You can modify the prompt by modifying `LABELS` in the demo notebook.

## Acknowledgements

This project builds on top of the official [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repository from IDEA-Research.

Many thanks to the GroundingDINO authors and maintainers for open-sourcing the model, codebase, pretrained checkpoints, and installation flow that make phrase-conditioned open-vocabulary detection practical to build on.

This repository uses GroundingDINO as the open-vocabulary detection backbone and adapts its detections for multi-object tracking.
