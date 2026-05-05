import os
import glob
import subprocess
import tqdm
import config
import sys

SMILE_BIN_DIR = os.path.abspath(os.path.join(config.PATH_TO_OPENSMILE, 'bin'))
SMILE_EXTRACT = os.path.join(SMILE_BIN_DIR, 'SMILExtract.exe')
CONFIG_FILE = os.path.join(config.PATH_TO_OPENSMILE, 'config', 'is09-13', 'IS13_ComParE.conf') 

if len(sys.argv) > 1:
    AUDIO_DIR = os.path.abspath(sys.argv[1])
else:
    AUDIO_DIR = os.path.abspath(r"D:\NCKH\MERTools\MER2023\dataset-process\audio")

SAVE_DIR = os.path.abspath(r"D:\NCKH\MERTools\MER2023\dataset-process\features\opensmile_is13")

if not os.path.exists(SAVE_DIR): 
    os.makedirs(SAVE_DIR)

my_env = os.environ.copy()
my_env["PATH"] = SMILE_BIN_DIR + os.pathsep + my_env["PATH"]

file_list = glob.glob(os.path.join(AUDIO_DIR, "*.wav")) + \
            glob.glob(os.path.join(AUDIO_DIR, "*.mp4")) + \
            glob.glob(os.path.join(AUDIO_DIR, "*.avi"))

for file_path in tqdm.tqdm(file_list):
    name = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = os.path.join(SAVE_DIR, name + ".csv")
    
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0: 
        continue
    
    cmd = [
        SMILE_EXTRACT,
        "-C", CONFIG_FILE,
        "-I", file_path,
        "-csvoutput", out_csv,
        "-noconsoleoutput", "1"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=my_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        subprocess.run(cmd, env=my_env) 
        break