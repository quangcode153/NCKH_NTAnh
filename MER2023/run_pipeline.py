import subprocess
import time
import sys

def auto_run_pipeline():
    scripts = [
        "extract_audio_features.py",
        "extract_visual_features.py",
        "train_multimodal_v3.py"
    ]

    print("="*55)
    print("BAT DAU CHAY DAY CHUYEN TU DONG")
    print("="*55)

    start_time = time.time()

    for step, script in enumerate(scripts, 1):
        print(f"\n[BUOC {step}/3] Dang khoi dong: {script} ...")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"HOAN THANH: {script}")
            
        except subprocess.CalledProcessError:
            print(f"\nLOI: Day chuyen dung tai {script}")
            return
            
        except FileNotFoundError:
            print(f"\nKHONG TIM THAY FILE: {script}")
            return

    end_time = time.time()
    minutes = int((end_time - start_time) // 60)
    seconds = int((end_time - start_time) % 60)
    
    print("\n" + "="*55)
    print("HOAN TAT TOAN BO QUY TRINH")
    print(f"Thoi gian: {minutes} phut {seconds} giay")
    print("="*55)

if __name__ == "__main__":
    auto_run_pipeline()