
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights 
import sys

sys.path.append('../../')
import config
from dataset import FaceDataset

def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in data_loader:
            images = images.cuda()
            embedding = model(images)
            embedding = embedding.squeeze() 
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    params = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    print('==> Preparing to extract ImageNet embeddings...')
    
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'imagenet_{params.feature_level[:3]}')

    if not os.path.exists(face_dir):
        print(f"❌ LỖI: Không tìm thấy thư mục: {face_dir}")
        print(f"👉 Vui lòng tạo thư mục này tại: D:\\NCKH\\MERTools\\MER2023\\dataset-process\\openface_face")
        sys.exit(1)

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    print("==> Loading ResNet18 model...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model = model.cuda()
    model = nn.Sequential(*list(model.children())[:-1]) 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vids = os.listdir(face_dir)
    if len(vids) == 0:
        print(f"⚠️ CẢNH BÁO: Thư mục '{face_dir}' đang rỗng. Không có video để xử lý!")
        sys.exit(0)

    print(f'Tìm thấy tổng cộng "{len(vids)}" thư mục khuôn mặt.')
    
    EMBEDDING_DIM = 512 
    for i, vid in enumerate(vids, 1):
        print(f"[{i}/{len(vids)}] Đang xử lý: {vid}")

        dataset = FaceDataset(vid, face_dir, transform=transform)
        if len(dataset) == 0:
            print(f"   - Bỏ qua: {vid} (Không có ảnh khuôn mặt)")
            continue
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True)
        embeddings, framenames = extract(data_loader, model)

        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes].squeeze()
        
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        
        if params.feature_level != 'FRAME' and len(embeddings.shape) == 2:
            embeddings = np.mean(embeddings, axis=0)
            
        np.save(csv_file, embeddings)
    
    print("✅ Hoàn thành trích xuất đặc trưng hình ảnh!")