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
        self.audio_net = nn.Sequential(
            nn.BatchNorm1d(6373), nn.Linear(6373, 1024), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(1024, 128), nn.ReLU()
        )
        self.visual_net = nn.Sequential(
            nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 6)
        )

    def forward(self, audio, visual):
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        combined = torch.cat((a, v), dim=1)
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

    print(f"Phat hien {len(video_files)} video. Dang xu ly...\n")

    model = FusionModelV3()
    try:
        model.load_state_dict(torch.load("ai_emotion_brain.pth", weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Loi: Khong tim thay file 'ai_emotion_brain.pth'.")
        return

    summary_report = []

    for video_name in video_files:
        print(f"Processing: {video_name}")
        base_name = os.path.splitext(video_name)[0]
        
        for f in os.listdir(SANDBOX_DIR):
            file_path = os.path.join(SANDBOX_DIR, f)
            if os.path.isfile(file_path): os.remove(file_path)

        shutil.copy(os.path.join(INPUT_DIR, video_name), os.path.join(SANDBOX_DIR, video_name))

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
                shutil.move(audio_csv, os.path.join(FEATURES_TEST_DIR, "audio_csv", f"{base_name}.csv"))
            if os.path.exists(visual_npy):
                shutil.move(visual_npy, os.path.join(FEATURES_TEST_DIR, "visual_npy", f"{base_name}.npy"))
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