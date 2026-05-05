import os
import glob
import subprocess
import tqdm
import config

video_dir = r"D:\NCKH\MERTools\MER2023\dataset-process\video"
audio_dir = r"D:\NCKH\MERTools\MER2023\dataset-process\audio"
ffmpeg_bin = config.PATH_TO_FFMPEG

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

videos = glob.glob(os.path.join(video_dir, "*"))
print(f"==> Tìm thấy {len(videos)} file. Bắt đầu tách âm thanh...")

for v_path in tqdm.tqdm(videos):
    name = os.path.splitext(os.path.basename(v_path))[0]
    out_path = os.path.join(audio_dir, name + ".wav")
    
    if os.path.exists(out_path):
        continue
    
    cmd = [ffmpeg_bin, "-y", "-i", v_path, "-ar", "16000", "-ac", "1", out_path]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"Lỗi tại file {name}: {e}")

print(f" HOÀN THÀNH! Bạn hãy kiểm tra thư mục: {audio_dir}")