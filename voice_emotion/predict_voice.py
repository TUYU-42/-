# predict_voice.py － 語音情緒辨識 + 給 Streamlit / GPT 用的封裝（與 train_voice_model_strong.py 對齊）

import os
import numpy as np
import librosa
import joblib

# =========================
# Config (要跟訓練時一致)
# =========================
SAMPLE_RATE = 16000
FIXED_SECONDS = 3.0
N_MFCC = 40

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "voice_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "voice_label_encoder.pkl")

# 載入已訓練好的模型與 LabelEncoder
clf = joblib.load(MODEL_PATH)           # pipeline: scaler + svm
label_encoder = joblib.load(ENCODER_PATH)

# =========================
# Audio utils (與訓練一致)
# =========================
def load_audio_fixed(path: str, sr: int, fixed_seconds: float, train: bool) -> np.ndarray:
    """
    與訓練版本一致：
    - librosa.load
    - trim 靜音
    - normalize
    - pad / crop
      - train=True: random crop
      - train=False: 中間 crop（推論用）
    """
    y, _ = librosa.load(path, sr=sr, mono=True)

    # trim：去掉前後長段靜音
    y, _ = librosa.effects.trim(y, top_db=25)

    # normalize：把音量尺度拉到一致
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak

    target_len = int(sr * fixed_seconds)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
        return y

    if train:
        start = np.random.randint(0, len(y) - target_len + 1)
    else:
        start = (len(y) - target_len) // 2

    return y[start:start + target_len]

def _stats(feat_2d: np.ndarray) -> np.ndarray:
    """(F, T) -> concat(mean(F), std(F)) -> (2F,)"""
    m = np.mean(feat_2d, axis=1)
    s = np.std(feat_2d, axis=1)
    return np.concatenate([m, s], axis=0)

def extract_features(file_path: str) -> np.ndarray:
    """
    回傳固定長度向量 (248,)
    必須與 train 的 extract_features 完全一致
    """
    path = file_path.strip().strip('"').strip("'")

    y = load_audio_fixed(path, sr=SAMPLE_RATE, fixed_seconds=FIXED_SECONDS, train=False)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)  # (40, T)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate([_stats(mfcc), _stats(d1), _stats(d2)], axis=0)  # 240

    zcr = librosa.feature.zero_crossing_rate(y)                         # (1, T)
    rms = librosa.feature.rms(y=y)                                      # (1, T)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)  # (1, T)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE) # (1, T)

    extra = np.array([
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms),
        np.mean(spec_cent), np.std(spec_cent),
        np.mean(spec_bw), np.std(spec_bw),
    ], dtype=np.float32)  # 8

    out = np.concatenate([feat, extra], axis=0).astype(np.float32)      # 248
    return out

# =========================
# Prediction
# =========================
def predict_voice_distribution(wav_path: str) -> np.ndarray:
    feat = extract_features(wav_path)          # (248,)
    X = feat.reshape(1, -1)                   # ✅ 變成 (1, 248)
    probs = clf.predict_proba(X)[0]
    return probs

def predict_voice_emotion(wav_path: str):
    probs = predict_voice_distribution(wav_path)
    idx = int(np.argmax(probs))
    label = label_encoder.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return label, conf, probs

def analyze_voice_for_gpt(wav_path: str) -> dict:
    label, conf, probs = predict_voice_emotion(wav_path)

    classes = label_encoder.classes_
    pairs = [(str(classes[i]), float(probs[i])) for i in range(len(classes))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    top3 = pairs[:3]

    top1_p = top3[0][1]
    top2_p = top3[1][1] if len(top3) > 1 else 0.0
    margin = top1_p - top2_p
    is_uncertain = (top1_p < 0.55) or (margin < 0.12)

    return {
        "main_label": label,
        "main_conf": float(top1_p),
        "top3": top3,
        "margin": float(margin),
        "is_uncertain": bool(is_uncertain),
    }

if __name__ == "__main__":
    test_wav = "test.wav"
    label, conf, probs = predict_voice_emotion(test_wav)
    print(f"Emotion: {label} | Confidence: {conf:.3f}")
    for cls, p in zip(label_encoder.classes_, probs):
        print(f"{cls:8s}: {p:.3f}")
