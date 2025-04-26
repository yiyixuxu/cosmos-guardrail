# cosmos-guardrail

A PyPI-installable package of NVIDIA's Cosmos Content Safety Guardrails  
(Checking text and blurring faces in video for unsafe content).

## Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # on Linux/macOS
   .venv\Scripts\activate.bat      # on Windows
   ```

2. Install the package:

   ```bash
   # From PyPI
   # pip install cosmos-guardrail
   pip install git+https://github.com/yiyixuxu/cosmos-guardrail.git

   # —or— for local development
   pip install -e .
   ```


## Quickstart

```python
from cosmos_guardrail import CosmosSafetyChecker
from cosmos_guardrail.utils import load_video
import numpy as np

prompt_input = "naked women"
video_input = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
)[:21]

# Convert list of PIL images to numpy array
frames = [np.array(frame) for frame in video_input]
video_input = np.stack(frames, axis=0)  # [T, H, W, C]

safety_checker = CosmosSafetyChecker()
print("Text safe?", safety_checker.check_text_safety(prompt_input))
checked_video = safety_checker.check_video_safety(video_input)
print("Video shape:", checked_video.shape if checked_video is not None else None)
```


## Credits

This package re-uses and lightly modifies code from NVIDIA's Cosmos projects:

- https://github.com/nvidia-cosmos/cosmos-transfer1  

All credit & copyright belong to NVIDIA (Apache-2.0).
