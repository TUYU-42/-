# predict_face.py － 臉部情緒辨識 + 給 GPT 用的封裝

import os
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = (48, 48)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_model.h5")

# 注意：順序要跟訓練時資料夾 class_names 一樣
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 載入模型
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img_path: str) -> np.ndarray:
    """
    讀入一張臉部圖片，轉成灰階 48x48，shape = (1, 48, 48, 1)
    支援含中文的 Windows 路徑。
    """
    path = img_path.strip().strip('"').strip("'")

    # 用 fromfile + imdecode 來支援中文路徑
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"讀取圖片失敗：{path}")

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)   # (48,48) -> (48,48,1)
    img = np.expand_dims(img, axis=0)    # -> (1,48,48,1)

    return img



def predict_face_distribution(img_path: str) -> np.ndarray:
    """
    回傳每個情緒的機率分布 (len = 7)
    """
    img = preprocess_image(img_path)
    preds = model.predict(img)[0]  # shape = (7,)
    return preds


def predict_face_emotion(img_path: str):
    """
    回傳：
    - 主情緒 label (str)
    - 信心度 (float)
    - 全部機率分布 (np.ndarray)
    """
    preds = predict_face_distribution(img_path)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf = float(preds[idx])
    return label, conf, preds


def analyze_face_for_gpt(img_path: str) -> dict:
    """
    給 main.py / GPT 使用的封裝。
    回傳 dict：
    {
      "main_label": "sad",
      "main_conf": 0.40,
      "top3": [("sad", 0.40), ("angry", 0.27), ("fear", 0.24)]
    }
    """
    label, conf, preds = predict_face_emotion(img_path)

    # 整理成 (label, prob) 並排序取前 3 個
    pairs = [(CLASS_NAMES[i], float(preds[i])) for i in range(len(CLASS_NAMES))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    top3 = pairs[:3]

    return {
        "main_label": label,
        "main_conf": conf,
        "top3": top3,
    }


if __name__ == "__main__":
    # 單獨測試用
    test_img = "test.jpg"  # 可改成你的圖片
    label, conf, preds = predict_face_emotion(test_img)

    print("Full distribution:")
    for name, p in zip(CLASS_NAMES, preds):
        print(f"{name:8s}: {p:.3f}")
    print(f"\nEmotion: {label} | Confidence: {conf:.3f}")
