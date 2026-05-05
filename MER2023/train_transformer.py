import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.join(BASE_DIR, 'dataset-release')
LABEL_FILES = glob.glob(os.path.join(RELEASE_DIR, '*label*.csv*'))

AUDIO_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'opensmile_is13')
VISUAL_DIR = os.path.join(BASE_DIR, 'dataset-process', 'features', 'vision_resnet')

EMOTION_MAP = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
    'worried': 4, 'fear': 4, 'surprise': 5, 'disgust': 5
}
CLASSES = ['neutral', 'happy', 'sad', 'angry', 'worried', 'surprise']

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
    exit()

df_master = pd.concat(all_labels, ignore_index=True).drop_duplicates(subset=['filename'])
print(f"Total samples: {len(df_master)}")

valid_data = []
for _, row in tqdm(df_master.iterrows(), total=df_master.shape[0]):
    raw_name = str(row['filename']).strip()
    label_str = str(row['label']).strip().lower()

    if label_str not in EMOTION_MAP:
        continue

    candidates = [
        raw_name,
        raw_name.replace('samplenew', 'sample'),
        raw_name.replace('sample', 'samplenew')
    ]

    final_name = next(
        (c for c in candidates if os.path.exists(os.path.join(VISUAL_DIR, c + '.npy'))),
        None
    )

    if final_name and os.path.exists(os.path.join(AUDIO_DIR, final_name + '.csv')):
        valid_data.append({
            'key': final_name,
            'label': EMOTION_MAP[label_str],
            'audio_path': os.path.join(AUDIO_DIR, final_name + '.csv'),
            'visual_path': os.path.join(VISUAL_DIR, final_name + '.npy')
        })

print(f"Valid samples: {len(valid_data)}")

class RealMultimodalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        try:
            audio_feat = pd.read_csv(item['audio_path'], sep=';').iloc[0, 2:].values.astype(np.float32)
        except:
            audio_feat = np.zeros(6373, dtype=np.float32)

        try:
            visual_feat = np.mean(np.load(item['visual_path']), axis=0).astype(np.float32)
        except:
            visual_feat = np.zeros(512, dtype=np.float32)

        return (
            torch.tensor(audio_feat),
            torch.tensor(visual_feat),
            torch.tensor(item['label'], dtype=torch.long)
        )

train_list, test_list = train_test_split(valid_data, test_size=0.2, random_state=42)

train_loader = DataLoader(RealMultimodalDataset(train_list), batch_size=32, shuffle=True)
test_loader = DataLoader(RealMultimodalDataset(test_list), batch_size=32, shuffle=False)

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        
        self.audio_proj = nn.Sequential(
            nn.BatchNorm1d(6373), nn.Linear(6373, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 128) 
        )
        self.visual_proj = nn.Sequential(
            nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 128) 
        )

        self.cross_att_v2a = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.5) 
        self.cross_att_a2v = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.5) 

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
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

model = FusionModel().to(device)

class_weights = torch.tensor([1.0, 1.2, 2.5, 1.5, 2.0, 4.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2) 

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

best_val_acc = 0.0

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for audio, visual, label in tqdm(train_loader, leave=False):
        audio, visual, label = audio.to(device), visual.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(audio, visual)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for audio, visual, label in test_loader:
            audio, visual, label = audio.to(device), visual.to(device), label.to(device)
            outputs = model(audio, visual)
            loss = criterion(outputs, label)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()
            
    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100 * val_correct / val_total
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "brain_transformer_best.pth")
        print(f"   -> [CẬP NHẬT] Kỷ lục mới! Đã lưu Best Model với Val Acc: {best_val_acc:.2f}%")

print("\n--- HOÀN TẤT HUẤN LUYỆN ---")
print("Đang tải lại Best Model để đánh giá và vẽ Ma trận nhầm lẫn...")
model.load_state_dict(torch.load("brain_transformer_best.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for audio, visual, label in test_loader:
        audio, visual, label = audio.to(device), visual.to(device), label.to(device)
        outputs = model(audio, visual)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

print(f"\nAccuracy (Best Model): {accuracy_score(all_labels, all_preds)*100:.2f}%")
print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

print("\nĐang tự động vẽ biểu đồ...")

epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], 'g-', label='Train Loss', marker='o', markersize=4)
plt.plot(epochs_range, history['val_loss'], 'r--', label='Val Loss', marker='x', markersize=4)
plt.title('Hội tụ Hàm Mất Mát (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', marker='o', markersize=4)
plt.plot(epochs_range, history['val_acc'], 'orange', linestyle='--', label='Val Acc', marker='x', markersize=4)
plt.title('Độ Chính Xác (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('2_training_curve_optimized.png', dpi=300)
plt.close()

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES, 
            annot_kws={"size": 12, "weight": "bold"})
plt.title(f'Ma trận nhầm lẫn (Confusion Matrix) - Best Val Acc: {best_val_acc:.2f}%', fontsize=14, pad=15)
plt.ylabel('Nhãn Thực tế (True Label)', fontsize=12, fontweight='bold')
plt.xlabel('Nhãn AI Dự đoán (Predicted Label)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('4_confusion_matrix_optimized.png', dpi=300)
plt.close()

print("Hoàn tất! Cùng kiểm tra 2 bức ảnh '2_training_curve_optimized.png' và '4_confusion_matrix_optimized.png' nhé.")