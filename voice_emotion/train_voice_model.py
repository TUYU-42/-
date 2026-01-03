# train_voice_model_strong.py
import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Paths / Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # voice_emotion 資料夾
TRAIN_DIR = os.path.join(BASE_DIR, "archive", "train")

SAMPLE_RATE = 16000
N_MFCC = 40
FIXED_SECONDS = 3.0   # 統一長度（秒）

MODEL_PATH = os.path.join(BASE_DIR, "voice_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "voice_label_encoder.pkl")

# =========================
# Audio utils
# =========================
def load_audio_fixed(path: str, sr: int, fixed_seconds: float, train: bool) -> np.ndarray:
    """
    讀音檔並統一長度：
    - 先 trim 靜音
    - normalize 音量
    - 長度不足補零
    - 長度足夠：train 時 random crop；eval 時取中間
    """
    y, _ = librosa.load(path, sr=sr, mono=True)

    # 去掉長段靜音（讓 crop 比較抓到有聲音的部分）
    y, _ = librosa.effects.trim(y, top_db=25)

    # normalize（避免不同錄音音量差太大）
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

# =========================
# Feature Extraction
# =========================
def extract_features(file_path: str, train: bool) -> np.ndarray:
    """
    強化特徵：
    - MFCC (40) + delta + delta2
    - 每個都取 mean/std
    - 加上 zcr/rms/spectral centroid/bw 的 mean/std
    """
    y = load_audio_fixed(file_path, sr=SAMPLE_RATE, fixed_seconds=FIXED_SECONDS, train=train)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)        # (40, T)
    d1   = librosa.feature.delta(mfcc)                                     # (40, T)
    d2   = librosa.feature.delta(mfcc, order=2)                            # (40, T)

    def stats(feat_2d):
        m = np.mean(feat_2d, axis=1)
        s = np.std(feat_2d, axis=1)
        return np.concatenate([m, s], axis=0)

    feat = np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0)     # 240

    zcr = librosa.feature.zero_crossing_rate(y)                            # (1, T)
    rms = librosa.feature.rms(y=y)                                         # (1, T)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)     # (1, T)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)    # (1, T)

    extra = np.array([
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms),
        np.mean(spec_cent), np.std(spec_cent),
        np.mean(spec_bw), np.std(spec_bw),
    ], dtype=np.float32)  # 8

    return np.concatenate([feat, extra], axis=0).astype(np.float32)        # 248

def load_dataset(root_dir: str, crops_per_file: int = 3):
    """
    每個音檔抽多段 random crop，變成更多訓練樣本
    """
    X, y = [], []
    for label_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                continue

            fpath = os.path.join(class_dir, fname)

            for _ in range(crops_per_file):
                try:
                    X.append(extract_features(fpath, train=True))
                    y.append(label_name)
                except Exception as e:
                    print(f"[WARN] 讀取失敗: {fpath} ({e})")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    return X, y

# =========================
# Train
# =========================
def main():
    print("=== 讀取訓練資料 ===")
    X, y = load_dataset(TRAIN_DIR, crops_per_file=3)
    print("資料形狀:", X.shape, y.shape)
    if len(X) == 0:
        raise RuntimeError("資料載入失敗：X 是空的，請確認 TRAIN_DIR 路徑與音檔副檔名。")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("類別:", le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

    param_grid = {
        "svm__C":     [0.5, 1, 2, 5, 10],
        "svm__gamma": ["scale", 0.01, 0.05, 0.1, 0.2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=== GridSearchCV 找最佳參數 ===")
    gs = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring="f1_macro", cv=cv, n_jobs=-1, verbose=2
    )
    gs.fit(X_train, y_train)

    print("最佳參數:", gs.best_params_)
    print("最佳 CV 分數(f1_macro):", gs.best_score_)

    best_model = gs.best_estimator_

    print("\n=== Hold-out 測試集評估 ===")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n=== 儲存模型與 LabelEncoder ===")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("已儲存:", MODEL_PATH, ENCODER_PATH)

if __name__ == "__main__":
    main()
