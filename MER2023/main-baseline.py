import os
import tqdm
import glob
import subprocess
import numpy as np
import pandas as pd
import config

def split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    video_list = glob.glob(os.path.join(video_root, '*'))
    print(f"==> Đang tách âm thanh từ {len(video_list)} video...")
    
    success = 0
    for video_path in tqdm.tqdm(video_list):
        videoname = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(save_root, videoname + '.wav')
        
        if os.path.exists(audio_path):
            continue
        
        cmd = [
            config.PATH_TO_FFMPEG,
            "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1",
            audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            success += 1
        except:
            pass

    print(f" Đã tạo {success} file .wav mới tại {save_root}")