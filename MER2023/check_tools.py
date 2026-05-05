import os
import config
import subprocess
import torch

print("SYSTEM CHECK MER2023")

video_dir = "./dataset-process/video"
if os.path.exists(video_dir):
    num_vids = len([f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
    print(f"video dir: {video_dir}")
    print(f"num videos: {num_vids}")
else:
    print(f"error: missing video dir {video_dir}")

ffmpeg_path = config.PATH_TO_FFMPEG
print(f"\nffmpeg path: {ffmpeg_path}")

if os.path.exists(ffmpeg_path):
    print("ffmpeg found")
    try:
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        print(f"ffmpeg ok: {result.stdout.splitlines()[0]}")
    except Exception as e:
        print(f"ffmpeg error: {e}")
else:
    print("ffmpeg not found")

print("\nGPU CHECK")
print(f"torch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
else:
    print("gpu not available (cpu mode)")