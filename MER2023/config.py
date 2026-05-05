import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = {
    'MER2023': os.path.join(BASE_DIR, 'dataset-process'),
}

PATH_TO_RAW_AUDIO = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
}

PATH_TO_RAW_FACE = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}

PATH_TO_TRANSCRIPTIONS = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription.csv'),
}

PATH_TO_FEATURES = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'features'),
}

PATH_TO_LABEL = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
}

PATH_TO_PRETRAINED_MODELS = os.path.join(BASE_DIR, 'tools')

PATH_TO_OPENSMILE = os.path.join(PATH_TO_PRETRAINED_MODELS, 'opensmile-3.0.2-windows-x86_64')

PATH_TO_FFMPEG = os.path.join(PATH_TO_PRETRAINED_MODELS, 'ffmpeg', 'ffmpeg.exe')

SAVED_ROOT = os.path.join(BASE_DIR, 'saved')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')