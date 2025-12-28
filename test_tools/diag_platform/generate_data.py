import os
import json
import numpy as np
import random
import datetime
import re

# ==========================================
# 설정 (Configuration)
# ==========================================
OUTPUT_DIR = "./dummy_data_sample"
NUM_SAMPLES = 50  # 생성할 파일 세트 수
SAMPLE_RATE = 51200
DURATION_SEC = 3.0
CHANNELS = 3
DATE_START = datetime.datetime(2023, 1, 1)

# ==========================================
# 유틸리티 함수
# ==========================================
def generate_random_sn(length=15):
    """15자리 알파뉴메릭 시리얼 번호 생성"""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))

def generate_random_date(start_date, day_range=30):
    """임의의 날짜 및 시간 생성"""
    random_days = random.randint(0, day_range)
    random_seconds = random.randint(0, 86400)
    dt = start_date + datetime.timedelta(days=random_days, seconds=random_seconds)
    return dt

def generate_filename(dt, sn, suffix):
    """파일명 규칙 생성: ^.*DADA_\d{8}_\d{6}__SW_(OK|NG)\.json"""
    # 예: FACTORY01_DADA_20231025_123000_A1B2C3D4E5F6G7H_SW_OK.json
    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M%S")
    result = random.choice(["OK", "NG"])
    prefix = random.choice(['a','b','c'])
    
    filename = f"{prefix}_DADA_{date_str}_{time_str}_{sn}_{suffix}_{result}.json"
    return filename, result

# ==========================================
# 데이터 생성 로직
# ==========================================
def create_dummy_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    for i in range(NUM_SAMPLES):
        # 1. 공통 메타데이터 생성
        dt = generate_random_date(DATE_START)
        sn = generate_random_sn()
        
        # --------------------------------------
        # 2. Raw Data (SW) 생성
        # --------------------------------------
        sw_filename, result = generate_filename(dt, sn, "SW")
        
        # 가짜 파형 데이터 생성 (Sine wave + Noise)
        num_points = int(SAMPLE_RATE * DURATION_SEC)
        t = np.linspace(0, DURATION_SEC, num_points)
        
        ch_data = {}
        for ch in range(1, CHANNELS + 1):
            freq = random.uniform(100, 1000)
            noise = np.random.normal(0, 0.1, num_points)
            # JSON 직렬화를 위해 list로 변환
            waveform = (np.sin(2 * np.pi * freq * t) + noise).astype(np.float32).tolist()
            ch_data[f"ch_{ch}"] = waveform

        # DEV_NM 설정 (Spin, Dryer, 기타)
        dev_nm_candidates = ['Spin','Dryer','Wash']
        dev_nm = random.choice(dev_nm_candidates)

        sw_json = {
            "header": {
                "DEV_NM": dev_nm,
                "Description": "Dummy Data for Testing",
                "Model_nm": None,
                "Physical_nm": None,
                "BitsPerSample": 32,
                "SamplesPerSecond": SAMPLE_RATE,
                "SoundInputChannels": CHANNELS
            },
            "ts": {
                "interval": 1.0 / SAMPLE_RATE,
                **ch_data  # ch_1, ch_2, ch_3...
            }
        }

        # SW 파일 저장
        sw_path = os.path.join(OUTPUT_DIR, sw_filename)
        with open(sw_path, 'w') as f:
            json.dump(sw_json, f) # 용량을 줄이려면 separators=(',', ':') 사용 가능

        # --------------------------------------
        # 3. Feature Data (SF) 생성 (SW와 Pair)
        # --------------------------------------
        # 20% 확률로 SF 파일 누락 시뮬레이션 (SW만 있는 경우 테스트용)
        if random.random() > 0.2: 
            sf_filename = sw_filename.replace("_SW_", "_SF_")
            
            sf_data = {
                "factory_location": "Busan_Factory_1",
                "factory_process": "Final_Inspection"
            }

            # Feature String 생성 (key=value&key=value 형태)
            for ch in range(1, CHANNELS + 1):
                # 49개의 랜덤 피쳐 생성
                feats = []
                for f_idx in range(49):
                    val = random.uniform(0, 10)
                    feats.append(f"F{f_idx}={val:.4f}")
                
                feat_str = "&".join(feats)
                sf_data[f"Ch{ch}_Feature"] = feat_str

            # SF 파일 저장
            sf_path = os.path.join(OUTPUT_DIR, sf_filename)
            with open(sf_path, 'w') as f:
                json.dump(sf_data, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} sample pairs...")

    print(f"\nCompleted! Generated data in '{OUTPUT_DIR}'")
    print(f"Sample SW Filename: {sw_filename}")

if __name__ == "__main__":
    create_dummy_data()