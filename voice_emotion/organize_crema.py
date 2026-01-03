import os
import shutil
import random

AUDIO_DIR = "archive/AudioWAV"   # CREMA-D audio folder
TRAIN_DIR = "archive/train"
TEST_DIR = "archive/test"

# 比例：80% train，20% test
SPLIT_RATIO = 0.8

# 情緒對照表
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def ensure_dirs():
    for base in [TRAIN_DIR, TEST_DIR]:
        for emo in emotion_map.values():
            os.makedirs(os.path.join(base, emo), exist_ok=True)

def parse_emotion(filename):
    """
    CREMA-D 格式：1001_DFA_ANG_XX.wav
    第3段是情緒代號
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    emo_code = parts[2]
    return emotion_map.get(emo_code, None)

def main():
    ensure_dirs()
    files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]

    random.shuffle(files)

    split_idx = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # move train files
    for f in train_files:
        emo = parse_emotion(f)
        if emo:
            src = os.path.join(AUDIO_DIR, f)
            dst = os.path.join(TRAIN_DIR, emo, f)
            shutil.copy(src, dst)

    # move test files
    for f in test_files:
        emo = parse_emotion(f)
        if emo:
            src = os.path.join(AUDIO_DIR, f)
            dst = os.path.join(TEST_DIR, emo, f)
            shutil.copy(src, dst)

    print("完成分類！")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    main()
