import os
import glob
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    VIDEO_DIR = os.path.abspath(sys.argv[1])
else:
    VIDEO_DIR = os.path.join(BASE_DIR, 'dataset-process', 'video')

SAVE_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'vision_resnet')

if not os.path.exists(VIDEO_DIR):
    sys.exit()

video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + \
              glob.glob(os.path.join(VIDEO_DIR, "*.avi")) + \
              glob.glob(os.path.join(VIDEO_DIR, "*.mov"))

if len(video_files) == 0:
    sys.exit()

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
except Exception:
    sys.exit()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for video_path in tqdm(video_files, unit="video"):
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_npy = os.path.join(SAVE_DIR, name + ".npy")

    if os.path.exists(out_npy) and os.path.getsize(out_npy) > 0:
        continue

    features = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        continue

    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    feature = feature_extractor(input_tensor)
                    features.append(feature.cpu().numpy().flatten())

            frame_count += 1

    except Exception:
        pass
    finally:
        cap.release()

    if len(features) > 0:
        np.save(out_npy, np.array(features))