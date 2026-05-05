import os
import shutil
import subprocess
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class FusionModelV3(nn.Module):
    def __init__(self):
        super(FusionModelV3, self).__init__()
        
        self.audio_proj = nn.Sequential(
            nn.BatchNorm1d(6373), nn.Linear(6373, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 128)
        )
        self.visual_proj = nn.Sequential(
            nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 128)
        )

        self.cross_att_v2a = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.3)
        self.cross_att_a2v = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.3)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self, audio, visual):
        a = self.audio_proj(audio).unsqueeze(1) 
        v = self.visual_proj(visual).unsqueeze(1) 

        att_v, _ = self.cross_att_v2a(query=v, key=a, value=a)
        att_a, _ = self.cross_att_a2v(query=a, key=v, value=v)

        att_v = att_v.squeeze(1)
        att_a = att_a.squeeze(1)

        combined = torch.cat((att_a, att_v), dim=1)
        return self.classifier(combined)

EMOTIONS = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Worried', 5: 'Surprise'}

def master_pipeline():
    INPUT_DIR = "videoinput"
    OUTPUT_DIR = "videoOutput"
    SANDBOX_DIR = "inference_sandbox"
    FEATURES_TEST_DIR = "featuresOutput"

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    os.makedirs(os.path.join(FEATURES_TEST_DIR, "audio_csv"), exist_ok=True)
    os.makedirs(os.path.join(FEATURES_TEST_DIR, "visual_npy"), exist_ok=True)

    video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp4', '.avi'))]

    if not video_files:
        print(f"Thu muc '{INPUT_DIR}' trong.")
        return

    print(f"Phat hien {len(video_files)} video. Dang xu ly theo kien truc Transformer...\n")

    model = FusionModelV3()
    try:
        model.load_state_dict(torch.load("brain_transformer.pth", weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Loi: Khong tim thay file 'brain_transformer.pth'. Ban can chay file train_transformer.py truoc.")
        return

    summary_report = []

    for video_name in video_files:
        print(f"Processing: {video_name}")
        base_name = os.path.splitext(video_name)[0]
        
        for f in os.listdir(SANDBOX_DIR):
            file_path = os.path.join(SANDBOX_DIR, f)
            if os.path.isfile(file_path): os.remove(file_path)

        video_sandbox_path = os.path.join(SANDBOX_DIR, video_name)
        shutil.copy(os.path.join(INPUT_DIR, video_name), video_sandbox_path)

        try:
            from moviepy.editor import VideoFileClip
            wav_path = os.path.join(SANDBOX_DIR, base_name + ".wav")
            video_clip = VideoFileClip(video_sandbox_path)
            video_clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
            video_clip.close()
        except Exception as e:
            print(f"Canh bao: Khong the trich xuat am thanh (.wav) cho {video_name}: {e}")

        try:
            subprocess.run([sys.executable, "extract_audio_features.py", SANDBOX_DIR])
            subprocess.run([sys.executable, "extract_visual_features.py", SANDBOX_DIR])

            audio_csv = f"dataset-process/features/opensmile_is13/{base_name}.csv"
            visual_npy = f"dataset-process/features/vision_resnet/{base_name}.npy"

            if os.path.exists(audio_csv):
                audio_feat = pd.read_csv(audio_csv, sep=';').iloc[0, 2:].values.astype(np.float32)
            else:
                audio_feat = np.zeros(6373, dtype=np.float32)

            if os.path.exists(visual_npy):
                visual_feat = np.mean(np.load(visual_npy), axis=0).astype(np.float32)
            else:
                visual_feat = np.zeros(512, dtype=np.float32)

            with torch.no_grad():
                output = model(torch.tensor(audio_feat).unsqueeze(0), torch.tensor(visual_feat).unsqueeze(0))
                _, pred = torch.max(output, 1)
            
            res = EMOTIONS[pred.item()]
            summary_report.append({"video": video_name, "status": "Success", "emotion": res})
            print(f"-> Result: {res}")

            if os.path.exists(audio_csv):
                shutil.copy(audio_csv, os.path.join(FEATURES_TEST_DIR, "audio_csv", f"{base_name}.csv"))
            if os.path.exists(visual_npy):
                shutil.copy(visual_npy, os.path.join(FEATURES_TEST_DIR, "visual_npy", f"{base_name}.npy"))
            shutil.move(os.path.join(INPUT_DIR, video_name), os.path.join(OUTPUT_DIR, video_name))

        except Exception as e:
            print(f"Loi xu ly video {video_name}: {e}")
            summary_report.append({"video": video_name, "status": "Error", "emotion": "N/A"})

    for f in os.listdir(SANDBOX_DIR):
        file_path = os.path.join(SANDBOX_DIR, f)
        if os.path.isfile(file_path): os.remove(file_path)

    print("\n" + "-"*50)
    print(f"{'VIDEO':<25} | {'RESULT'}")
    print("-" * 50)
    for row in summary_report:
        print(f"{row['video'][:24]:<25} | {row['emotion']}")
    print("-" * 50)

if __name__ == "__main__":
    master_pipeline()