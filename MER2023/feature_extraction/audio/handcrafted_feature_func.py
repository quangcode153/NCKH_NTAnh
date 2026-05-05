
import os
import subprocess
import glob
import sys
import numpy as np
import librosa 

import sys
sys.path.append('../../')
from config import PATH_TO_OPENSMILE

class OPENSMILE(object):
    def __init__(self, path):
        self.name = 'opensmile'
        self.path = path
        if sys.platform == 'win32':
            self.OPENSMILE_EXE = os.path.join(self.path, r'bin/Win32/SMILExtract_Release.exe')
        else: 
            self.OPENSMILE_EXE = os.path.join(self.path, r'bin/linux_x64_standalone_static/SMILExtract')

        self.eGeMAPS_CFG_FILE = os.path.join(self.path, r'config/gemaps/eGeMAPSv01a.conf')
        
        self.IS09_Emo_CFG_FILE = os.path.join(self.path, r'config/IS09_emotion.conf')
        
        self.IS10_Emo_CFG_FILE = os.path.join(self.path, r'config/IS10_paraling.conf')
        
        self.IS13_ComParE_CFG_FILE = os.path.join(self.path, r'config/IS13_ComParE.conf')

        self.FrameModeFunctionals_CFG_FILE = os.path.join(self.path, r'config/shared/FrameModeFunctionals.conf.inc')

    def _verify_feature_level(self, feature_level):
        assert feature_level.upper() in ['FRAME', 'UTTERANCE'], ("Invalid acoustic feature level: %s!" % feature_level)
        return feature_level.upper()

    def _get_cfg_file(self, feature_set):
        if feature_set == 'IS09':
            return self.IS09_Emo_CFG_FILE
        elif feature_set == 'IS10':
            return self.IS10_Emo_CFG_FILE
        elif feature_set == 'IS13':
            return self.IS13_ComParE_CFG_FILE
        elif feature_set == 'eGeMAPS':
            return self.eGeMAPS_CFG_FILE
        else:
            raise Exception(f'Error: not Supported feature set("{feature_set}")!')

    def _generate_frame_mode_functionals_cfg_file(self, frame_mode_functionals_cfg):
        frame_size, frame_step = frame_mode_functionals_cfg
        prefix, ext = os.path.splitext(self.FrameModeFunctionals_CFG_FILE)
        new_frame_mode_functionals_cfg_file = f'{prefix}_{frame_size}_{frame_step}{ext}'
        if not os.path.exists(new_frame_mode_functionals_cfg_file):
            with open(new_frame_mode_functionals_cfg_file, 'w') as f:
                f.write('\n')
                f.write(';; set this to override the previously included BufferModeRb.conf.inc (in GeMAPS sets only) for the functionals\n')
                f.write(';; or (in all other sets), define the RbConf here to save memory on the functionals levels:\n')
                f.write(';writer.levelconf.growDyn = 0\n')
                f.write(';writer.levelconf.isRb = 1\n')
                f.write(';writer.levelconf.nT = 5\n')
                f.write('\n')
                f.write('frameMode = fixed\n')
                f.write(f'frameSize = {frame_size}\n')
                f.write(f'frameStep = {frame_step}\n')
                f.write('frameCenterSpecial = left\n')
        return new_frame_mode_functionals_cfg_file

    def extract_acoustic_feature(self,
                                 input_file,
                                 output_file,
                                 feature_set,
                                 feauture_level,
                                 frame_mode_functionals_param=None,
                                 del_output_file=False):
        
        cfg_file = self._get_cfg_file(feature_set)
        
        level = self._verify_feature_level(feauture_level)
        
        if level == 'FRAME':  
            
            template = '"{}" -C "{}" -I "{}" -instname "{}" -lldcsvoutput "{}" -noconsoleoutput'  
            cmd = template.format(self.OPENSMILE_EXE, cfg_file, input_file, input_file, output_file)
        else:  
            
            if frame_mode_functionals_param is not None:
                frame_mode_functionals_cfg_file = self._generate_frame_mode_functionals_cfg_file(
                    frame_mode_functionals_param)
                
                template = '"{}" -C "{}" -I "{}" -instname "{}" -csvoutput "{}" -appendcsv 0 -frameModeFunctionalsConf "{}" -noconsoleoutput'  
                cmd = template.format(self.OPENSMILE_EXE, cfg_file, input_file, input_file, output_file,
                                      frame_mode_functionals_cfg_file)
            else:
                template = '"{}" -C "{}" -I "{}" -instname "{}" -csvoutput "{}" -appendcsv 0 -noconsoleoutput'  
                cmd = template.format(self.OPENSMILE_EXE, cfg_file, input_file, input_file, output_file)
        
        subprocess.call(cmd, shell=True) 
        
        acoustic_feature = self.parse_acoustic_feature_csv_file(output_file, level=level)
        
        if del_output_file == True:
            os.remove(output_file)

        return acoustic_feature

    def parse_acoustic_feature_csv_file(self, csv_file, level):
        acoustic_feature = []
        with open(csv_file, 'r') as f:
            for line in f.readlines()[1:]:  
                line = line.strip().split(';')
                line_feature = [float(val) for val in line[2:]]  
                acoustic_feature.append(line_feature) 

        return np.array(acoustic_feature)

class Librosa:
    def __init__(self):
        self.name = 'librosa'

    @staticmethod
    def extract_acoustic_feature(input_file, feature_set, frame_size=0.025, frame_step=0.010, **kwargs):
        if feature_set == 'mel_spec':
            features = Librosa.extract_mel_spectrogram(input_file, frame_size=frame_size, frame_step=frame_step)
        elif feature_set == 'mfcc':
            features = Librosa.extract_mfcc(input_file, frame_size=frame_size, frame_step=frame_step)
        else:
            raise Exception(f'Not supported feature set "{feature_set}" for audio feature extractor "Librosa".')
        return features

    @staticmethod
    def extract_mel_spectrogram(input_file, frame_size=0.025, frame_step=0.010,
                                n_mels=128, n_fft=2048, log_mel=False, delta=False):
        y, sr = librosa.load(input_file)
        win_length, hop_length = int(frame_size * sr), int(frame_step * sr)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win_length, hop_length=hop_length,
                                                  n_fft=n_fft, n_mels=n_mels) 
        if log_mel:
            
           mel_spec = librosa.power_to_db(mel_spec)
        if delta:
            mel_spec_delta = librosa.feature.delta(mel_spec)
            mel_spec_delta_2 = librosa.feature.delta(mel_spec_delta)
            mel_spec = np.vstack((mel_spec, mel_spec_delta, mel_spec_delta_2))
        mel_spec = mel_spec.transpose()
        return mel_spec

    @staticmethod
    def extract_mfcc(input_file, frame_size=0.025, frame_step=0.010,
                     n_mfcc=40, n_mels=128, n_fft=2048, delta=True):
        y, sr = librosa.load(input_file)
        win_length, hop_length = int(frame_size * sr), int(frame_step * sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, win_length=win_length, hop_length=hop_length,
                                    n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)
        
        if delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_2 = librosa.feature.delta(mfcc_delta)
            mfcc = np.vstack((mfcc, mfcc_delta, mfcc_delta_2))
        mfcc = mfcc.transpose()
        return mfcc

if __name__ == "__main__":
    input_file = os.path.join(PATH_TO_OPENSMILE, 'example-audio/opensmile.wav')
    output_file = os.path.join(PATH_TO_OPENSMILE, 'example-audio/test_utt.txt')
    myOpensmile = OPENSMILE(PATH_TO_OPENSMILE)
    feature = myOpensmile.extract_acoustic_feature(input_file=input_file,
                                                   output_file=output_file,
                                                   feature_set='IS09',
                                                   feauture_level='UTTERANCE',
                                                   frame_mode_functionals_param=(0.2,0.1)) 
    print(feature.shape)