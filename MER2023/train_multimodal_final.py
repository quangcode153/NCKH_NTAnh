import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.join(BASE_DIR, 'dataset-release')

LABEL_FILES = glob.glob(os.path.join(RELEASE_DIR, '*label*.csv*'))

AUDIO_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'opensmile_is13')
VISUAL_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'vision_resnet')

EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'worried': 4,
    'fear': 4,
    'surprise': 5,
    'disgust': 5
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device.type}")

all_labels = []

for file_path in LABEL_FILES:
    try:
        df = pd.read_csv(file_path)

        if 'name' not in df.columns:
            df = pd.read_csv(file_path, header=None)
            df.columns = ['name', 'discrete'] + list(df.columns[2:])

        df = df.rename(columns={'name': 'filename', 'discrete': 'label'})

        if 'filename' in df.columns and 'label' in df.columns:
            all_labels.append(df[['filename', 'label']])
    except:
        pass

if not all_labels:
    print("No label file found")
    exit()

df_master = pd.concat(all_labels, ignore_index=True)
df_master = df_master.drop_duplicates(subset=['filename'])

print(f"Total samples: {len(df_master)}")

valid_data = []
missing_count = 0

for _, row in tqdm(df_master.iterrows(), total=len(df_master)):
    raw_name = str(row['filename']).strip()
    label_str = str(row['label']).strip().lower()

    if label_str not in EMOTION_MAP:
        continue

    label_id = EMOTION_MAP[label_str]

    candidates = [
        raw_name,
        raw_name.replace('samplenew', 'sample'),
        raw_name.replace('sample', 'samplenew')
    ]

    final_name = None
    for cand in candidates:
        if os.path.exists(os.path.join(VISUAL_DIR, cand + '.npy')):
            final_name = cand
            break

    if final_name and os.path.exists(os.path.join(AUDIO_DIR, final_name + '.csv')):
        valid_data.append({
            'key': final_name,
            'label': label_id,
            'audio_path': os.path.join(AUDIO_DIR, final_name + '.csv'),
            'visual_path': os.path.join(VISUAL_DIR, final_name + '.npy')
        })
    else:
        missing_count += 1

print(f"Valid samples: {len(valid_data)}")
print(f"Missing samples: {missing_count}")

class RealMultimodalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        try:
            df = pd.read_csv(item['audio_path'], sep=';')
            audio_feat = df.iloc[0, 2:].values.astype(np.float32)
        except:
            audio_feat = np.zeros(6373, dtype=np.float32)

        try:
            visual_data = np.load(item['visual_path'])
            visual_feat = np.mean(visual_data, axis=0).astype(np.float32)
        except:
            visual_feat = np.zeros(512, dtype=np.float32)

        return (
            torch.tensor(audio_feat),
            torch.tensor(visual_feat),
            torch.tensor(item['label'], dtype=torch.long)
        )

if len(valid_data) == 0:
    print("No valid data")
    exit()

train_list, test_list = train_test_split(valid_data, test_size=0.2, random_state=42)

train_loader = DataLoader(RealMultimodalDataset(train_list), batch_size=32, shuffle=True)
test_loader = DataLoader(RealMultimodalDataset(test_list), batch_size=32)

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.audio_net = nn.Sequential(
            nn.Linear(6373, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU()
        )
        self.visual_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, audio, visual):
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        return self.classifier(torch.cat((a, v), dim=1))

model = FusionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for audio, visual, label in tqdm(train_loader, leave=False):
        audio, visual, label = audio.to(device), visual.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(audio, visual)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={100*correct/total:.2f}%")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for audio, visual, label in test_loader:
        audio, visual = audio.to(device), visual.to(device)
        outputs = model(audio, visual)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.numpy())

if all_labels:
    print(f"Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
    print(classification_report(all_labels, all_preds))
else:
    print("No test data")