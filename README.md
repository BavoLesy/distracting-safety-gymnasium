<!--
Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
Copyright 2025 IDLab, University of Antwerp - imec. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This documentation has been created by IDLab, University of Antwerp - imec to document
the visual distraction suite for Safety Gymnasium environments.
Built upon the Safety Gymnasium framework by OmniSafe Team.
-->

# Distracting Safety Gymnasium

A comprehensive visual distraction suite for Safety Gymnasium environments, designed to evaluate the robustness of vision-based reinforcement learning agents under challenging visual conditions.

## Overview

Distracting Safety Gymnasium extends [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) with rich static or dynamic visual distractions including video backgrounds and color variations of all objects in the environment.


Our work is heavily inspired by the [Distracting Control Suite](https://github.com/google-research/google-research/blob/master/distracting_control/README.md) but specifically designed for safe navigation environments.


## Features
### Video Backgrounds: Replace static skybox with real-world video sequences, currently supports the DAVIS-2017 video dataset but other videos can also be used by changing the dataset path. Currently, the dynamic mode lowers the simulation FPS by around a factor of 4, significantely increasing training time. Future improvements are planned to speed up the dynamic mode.

### Color Distractions: Modify colors of environment objects at episode start or at each step. Objects can be filtered by type or by name.


## Installation

### Prerequisites

Safety Gymnasium uses the MuJoCo physics engine, so this must be installed first, see the [MuJoCo installation guide](https://mujoco.readthedocs.io/en/latest/programming/#getting-started) for more details.
Then, install the Distracting Safety Gymnasium package:
```bash
# Core dependencies
git clone https://github.com/BavoLesy/Distracting-Safety-Gymnasium.git
cd Distracting-Safety-Gymnasium
pip install -e .  
```

### DAVIS Dataset Setup
For the video backgrounds, the DAVIS-2017 dataset is used. This can be downloaded from the [DAVIS website](https://davischallenge.org/).
**Download DAVIS-2017 Dataset**
   ```bash
   # Visit https://davischallenge.org/ to download
   # Or use direct download (if available):
   wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
   unzip DAVIS-2017-trainval-480p.zip
   ```


## Configuration

### Basic Video Backgrounds

```yaml
# Static video backgrounds
static_video_config:
  env.safetygym.video_background_path: '/path/to/DAVIS/JPEGImages/480p'
  env.safetygym.video_alpha: 0.7
  env.safetygym.video_dynamic: False
  env.safetygym.video_dataset: 'train'
  env.safetygym.num_videos: 10

# Dynamic video backgrounds
dynamic_video_config:
  env.safetygym.video_background_path: '/path/to/DAVIS/JPEGImages/480p'
  env.safetygym.video_alpha: 0.8
  env.safetygym.video_dynamic: True
  env.safetygym.video_dataset: 'val'
  env.safetygym.num_videos: 5
```

### Color Distractions

```yaml
# Static color changes
static_color_config:
  env.safetygym.change_geoms_color: 'static'
  env.safetygym.beta_rgb: 0.3
  env.safetygym.object_filter: 'all'

# Dynamic color changes
dynamic_color_config:
  env.safetygym.change_geoms_color: 'dynamic'
  env.safetygym.beta_rgb: 0.5
  env.safetygym.object_filter: ['hazards', 'vases', 'walls']
```

### Combined Distractions

```yaml
# Maximum distraction configuration
max_distraction_config:
  # Video backgrounds
  env.safetygym.video_background_path: '/path/to/DAVIS/JPEGImages/480p'
  env.safetygym.video_alpha: 0.6
  env.safetygym.video_dynamic: True
  env.safetygym.video_dataset: 'train'
  env.safetygym.num_videos: 20
  
  # Color distractions
  env.safetygym.change_geoms_color: 'dynamic'
  env.safetygym.beta_rgb: 0.7
  env.safetygym.object_filter: 'all'
```

## Parameters Reference

### Video Background Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_background_path` | `str` | `None` | Path to DAVIS JPEGImages directory |
| `video_alpha` | `float` | `0.7` | Blending factor [0,1] (0=original, 1=pure video) |
| `video_dynamic` | `bool` | `True` | Whether to animate through video frames |
| `video_dataset` | `str/list` | `'train'` | Dataset split ('train', 'val') or list of video names |
| `num_videos` | `int` | `10` | Maximum number of videos to load |

### Color Distraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `change_geoms_color` | `str` | `'none'` | Color change mode ('none', 'static', 'dynamic') |
| `beta_rgb` | `float` | `0.0` | Color variation intensity [0,1] |
| `object_filter` | `str/list` | `'all'` | Objects to modify ('all' or list of object types) |

### Available Object Filters

- `'all'`: All environment objects
- `'hazards'`: Hazardous objects with costs
- `'walls'`: Wall and barrier objects
- `'vases'`: Vase objects specifically
- `'goals'`: Goal objects
- `'agents'`: Agent bodies
- Custom list: `['hazards', 'walls', 'goals']`

## Usage Examples

### Training with Distractions

```python
import safety_gymnasium

# Create environment with video backgrounds
env = safety_gymnasium.make(
    'SafetyPointGoal1-v0',
    render_mode='rgb_array',
    config={
        'video_background_path': '/path/to/DAVIS/JPEGImages/480p',
        'video_alpha': 0.7,
        'video_dynamic': True,
        'video_dataset': 'train',
        'num_videos': 10,
        'change_geoms_color': 'dynamic',
        'beta_rgb': 0.5,
        'object_filter': ['hazards', 'walls']
    }
)

# Standard RL loop
obs, info = env.reset()
for step in range(1000):
    action = policy(obs)  # Your policy here
    obs, reward, cost, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Evaluation Configurations

```yaml
# Easy distraction level
easy_distraction:
  env.safetygym.video_alpha: 0.3
  env.safetygym.video_dynamic: False
  env.safetygym.beta_rgb: 0.2
  env.safetygym.change_geoms_color: 'static'

# Medium distraction level  
medium_distraction:
  env.safetygym.video_alpha: 0.6
  env.safetygym.video_dynamic: True
  env.safetygym.beta_rgb: 0.4
  env.safetygym.change_geoms_color: 'dynamic'

# Hard distraction level
hard_distraction:
  env.safetygym.video_alpha: 0.9
  env.safetygym.video_dynamic: True
  env.safetygym.beta_rgb: 0.8
  env.safetygym.change_geoms_color: 'dynamic'
  env.safetygym.object_filter: 'all'
```


### Supported Environments
- All Safety Gymnasium navigation environments:
  - `SafetyPointGoal1-v0`, `SafetyPointGoal2-v0`
  - `SafetyCarGoal1-v0`, `SafetyCarGoal2-v0`
  - `SafetyPointButton1-v0`, `SafetyPointButton2-v0`



## Citation

If you use this distraction suite in your research, please cite:

```bibtex
@misc{distracting_safetygym,
  title={...},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/distracting-safety-gym}
}
```

## Related Work

- [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control)
- [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- [DAVIS Challenge](https://davischallenge.org/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
The original Safety Gymnasium framework is Copyright 2022-2023 OmniSafe Team.
Visual distraction extensions are Copyright 2025 IDLab, University of Antwerp - imec.