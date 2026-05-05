import streamlit as st
import os
import subprocess
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class FusionModelV3(nn.Module):
    def __init__(self):
        super(FusionModelV3, self).__init__()
        self.audio_proj = nn.Sequential(nn.BatchNorm1d(6373), nn.Linear(6373, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128))
        self.visual_proj = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128))
        self.cross_att_v2a = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.3)
        self.cross_att_a2v = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(nn.Linear(128 * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 6))

    def forward(self, audio, visual):
        a = self.audio_proj(audio).unsqueeze(1) 
        v = self.visual_proj(visual).unsqueeze(1) 
        att_v, _ = self.cross_att_v2a(query=v, key=a, value=a)
        att_a, _ = self.cross_att_a2v(query=a, key=v, value=v)
        att_v = att_v.squeeze(1)
        att_a = att_a.squeeze(1)
        combined = torch.cat((att_a, att_v), dim=1)
        return self.classifier(combined)

EMOTIONS = {0: 'Neutral (Bình thường) 😐', 1: 'Happy (Vui vẻ) 😄', 2: 'Sad (Buồn) 😢', 3: 'Angry (Tức giận) 😠', 4: 'Worried (Lo lắng) 😟', 5: 'Surprise (Ngạc nhiên) 😲'}

st.set_page_config(page_title="Hệ thống MER", page_icon="🎭", layout="wide")

st.markdown("<h1 style='text-align: center;'> CHẨN ĐOÁN CẢM XÚC VIDEO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Hệ thống AI nhận diện biểu cảm khuôn mặt và giọng nói bằng kiến trúc Multimodal Transformer</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1.2], gap="large") 

with col1:
    st.markdown("### 📥 Tải video lên đây")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn video (.mp4, .avi)", type=["mp4", "avi"], label_visibility="collapsed")
    
    analyze_button = st.button(" Phân tích ngay", use_container_width=True, type="primary")

with col2:
    st.markdown("###  Vùng phát video")
    video_placeholder = st.empty() 
    
    st.markdown("###  Chi tiết kết quả")
    result_placeholder = st.empty() 

    if uploaded_file is not None:
        video_placeholder.video(uploaded_file)
    else:
        video_placeholder.info("Video của bạn sẽ hiển thị ở đây sau khi tải lên.")
        result_placeholder.info("Kết quả phân tích sẽ xuất hiện ở đây.")

if analyze_button:
    if uploaded_file is None:
        with col1:
            st.error(" Vui lòng tải video lên trước khi phân tích!")
    else:
        with col2: 
            with st.spinner('AI đang bóc tách âm thanh và hình ảnh... Vui lòng đợi!'):
                try:
                    
                    SANDBOX_DIR = "inference_sandbox"
                    os.makedirs(SANDBOX_DIR, exist_ok=True)
                    for f in os.listdir(SANDBOX_DIR):
                        os.remove(os.path.join(SANDBOX_DIR, f))
                        
                    video_path = os.path.join(SANDBOX_DIR, uploaded_file.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    
                    from moviepy.editor import VideoFileClip
                    wav_path = os.path.join(SANDBOX_DIR, base_name + ".wav")
                    try:
                        video_clip = VideoFileClip(video_path)
                        video_clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
                        video_clip.close()
                    except Exception as e:
                        st.warning("Hệ thống không tìm thấy luồng âm thanh rõ ràng trong video này.")

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

                    model = FusionModelV3()
                    model.load_state_dict(torch.load("brain_transformer.pth", weights_only=True))
                    model.eval()
                    
                    with torch.no_grad():
                        output = model(torch.tensor(audio_feat).unsqueeze(0), torch.tensor(visual_feat).unsqueeze(0))
                        _, pred = torch.max(output, 1)
                    
                    res = EMOTIONS[pred.item()]
                    
                    result_placeholder.success(f"### Kết quả chẩn đoán: {res}")
                    st.balloons()
                    
                except Exception as e:
                    result_placeholder.error(f"Có lỗi hệ thống xảy ra: {e}")