# from diffusers.pipelines.cosmos import CosmosSafetyChecker
from cosmos_guardrail import CosmosSafetyChecker 
from cosmos_guardrail.utils import load_video
import numpy as np
# from diffusers.pipelines.video_processor import VideoProcessor

# Example text prompt to check
prompt_input = "naked women"

# Load sample video
video_input = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
)[:21]

# Convert list of PIL images to numpy array
frames = [np.array(frame) for frame in video_input]
video_input = np.stack(frames, axis=0)  # Shape: [T, H, W, C]

# Initialize the safety checker
safety_checker = CosmosSafetyChecker()

# Check text safety
is_text_safe = safety_checker.check_text_safety(prompt_input)

# Check video safety and apply face blurring
checked_video = safety_checker.check_video_safety(video_input)

# Print results
print(f"Is text safe: {is_text_safe}")
print(f"Checked video shape: {checked_video.shape if checked_video is not None else None}")


