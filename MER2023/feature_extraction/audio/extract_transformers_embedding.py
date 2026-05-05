import os
import time
import glob
import torch
import argparse
import numpy as np
import soundfile as sf

from transformers import Wav2Vec2FeatureExtractor 

import sys
sys.path.append('../../')
import config

HUBERT_BASE_CHINESE = 'chinese-hubert-base' 
HUBERT_LARGE_CHINESE = 'chinese-hubert-large' 
WAV2VEC2_BASE_CHINESE = 'chinese-wav2vec2-base' 
WAV2VEC2_LARGE_CHINESE = 'chinese-wav2vec2-large' 

WAV2VEC2_BASE = 'wav2vec2-base-960h' 
WAV2VEC2_LARGE = 'wav2vec2-large-960h' 

def extract(model_name, audio_files, save_dir, feature_level, layer_ids=None, gpu=None):

    start_time = time.time()

    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
    if model_name.find('hubert') != -1:
        from transformers import HubertModel
        model = HubertModel.from_pretrained(model_file)
    elif model_name.find('wav2vec') != -1:
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(model_file)

    if gpu != -1:
        device = torch.device(f'cuda:{gpu}')
        model.to(device)
    model.eval()

    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        samples, sr = sf.read(audio_file)
        input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values
        if gpu != -1: input_values = input_values.to(device)

        with torch.no_grad():
            
            hidden_states = model(input_values, output_hidden_states=True).hidden_states 
            feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)  
            assert feature.shape[0] == 1
            feature = feature[0].detach().squeeze().cpu().numpy() 

        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if feature_level == 'UTTERANCE':
            feature = np.array(feature).squeeze() 
            if len(feature.shape) != 1: 
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
        else:
            np.save(csv_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu')
    parser.add_argument('--model_name', type=str, default='opensmile', help='name of feature extractor')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    layer_ids = [-1]
    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    dir_name = args.model_name if len(layer_ids) == 1 else f'{args.model_name}-{len(layer_ids)}'
    dir_name = f'{dir_name}-{args.feature_level[:3]}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif args.overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')
    
    extract(args.model_name, audio_files, save_dir, args.feature_level, layer_ids=layer_ids, gpu=args.gpu)
