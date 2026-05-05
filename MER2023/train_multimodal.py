import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'opensmile_is13')
VISUAL_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'vision_resnet')
LABEL_PATH = os.path.join(BASE_DIR, 'dataset-process', 'label-6way.npz')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device.type}")

class MultimodalDataset(Dataset):
    def __init__(self, keys, labels):
        self.keys = keys
        self.labels = labels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        label = self.labels[idx]

        audio_path = os.path.join(AUDIO_DIR, key + '.csv')
        visual_path = os.path.join(VISUAL_DIR, key + '.npy')

        try:
            df = pd.read_csv(audio_path, sep=';')
            audio_feat = df.iloc[0, 2:].values.astype(np.float32)
        except:
            audio_feat = np.zeros(6373, dtype=np.float32)

        try:
            visual_feat = np.mean(np.load(visual_path), axis=0).astype(np.float32)
        except:
            visual_feat = np.zeros(512, dtype=np.float32)

        return (
            torch.tensor(audio_feat),
            torch.tensor(visual_feat),
            torch.tensor(label, dtype=torch.long)
        )

all_csv = set(os.path.splitext(f)[0] for f in os.listdir(AUDIO_DIR) if f.endswith('.csv'))
all_npy = set(os.path.splitext(f)[0] for f in os.listdir(VISUAL_DIR) if f.endswith('.npy'))

keys = list(all_csv.intersection(all_npy))
print(f"Samples: {len(keys)}")

labels = np.random.randint(0, 6, size=len(keys))

train_keys, test_keys, train_labels, test_labels = train_test_split(
    keys, labels, test_size=0.2, random_state=42
)

train_loader = DataLoader(MultimodalDataset(train_keys, train_labels), batch_size=32, shuffle=True)
test_loader = DataLoader(MultimodalDataset(test_keys, test_labels), batch_size=32, shuffle=False)

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_net = nn.Sequential(
            nn.Linear(6373, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
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
        x = torch.cat((a, v), dim=1)
        return self.classifier(x)

model = FusionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

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

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={acc:.2f}%")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for audio, visual, label in test_loader:
        audio, visual, label = audio.to(device), visual.to(device), label.to(device)
        outputs = model(audio, visual)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

print(f"Test accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")