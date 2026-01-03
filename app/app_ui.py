import os
import sys
import csv
import io
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display  # 用來畫 mel-spectrogram

# =========================
# ✅ 修 WinError 2：強制提供 ffmpeg/ffprobe
# =========================
from pydub import AudioSegment
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_DIR = os.path.dirname(FFMPEG_EXE)

# 1) pydub 可能用這些欄位（不同版本）
AudioSegment.converter = FFMPEG_EXE
AudioSegment.ffmpeg = FFMPEG_EXE
AudioSegment.ffprobe = FFMPEG_EXE

# 2) pydub.utils.which("ffprobe") 會看 PATH，所以保險把資料夾塞到 PATH 最前面
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# 3) Debug：先印出來，確定真的有路徑（成功後可刪）
print("FFMPEG_EXE =", FFMPEG_EXE)
print("FFMPEG_DIR =", FFMPEG_DIR)

# ✅ Streamlit 錄音元件（輸出 WAV bytes）
from audiorecorder import audiorecorder


# ====== 可選：支援 iPhone HEIC ======
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

# ====== 人臉偵測（OpenCV）=====
try:
    import cv2
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False

# 讓 Python 找得到 face_emotion / voice_emotion / main
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from face_emotion.predict_face import analyze_face_for_gpt
from voice_emotion.predict_voice import analyze_voice_for_gpt
from main import call_gpt_api


# ============================
# 多模態 Fusion
# ============================

POSITIVE = {"happy", "surprise"}
NEGATIVE = {"sad", "angry", "fear", "disgust"}
NEUTRAL = {"neutral"}


def fusion_emotion(face_info, voice_info):
    f_label = face_info["main_label"]
    v_label = voice_info["main_label"]
    f_conf = face_info["main_conf"]
    v_conf = voice_info["main_conf"]

    # 信心較高者為主
    # 語音不確定 → 不要讓它蓋過臉
    if voice_info.get("is_uncertain", False):
        fusion_label = f_label
    else:
        fusion_label = f_label if f_conf >= v_conf else v_label

    def group(lbl):
        if lbl in POSITIVE:
            return "positive"
        if lbl in NEGATIVE:
            return "negative"
        if lbl in NEUTRAL:
            return "neutral"
        return "other"

    f_group, v_group = group(f_label), group(v_label)
    conflict = False

    if (f_group == "positive" and v_group == "negative") or \
       (f_group == "negative" and v_group == "positive"):
        conflict = True
        comment = "臉部與語音情緒方向相反，可能存在情緒落差。"
    elif f_label != v_label and f_conf >= 0.5 and v_conf >= 0.5:
        conflict = True
        comment = "兩邊模型都很有信心，但主情緒不同，可能是混合情緒。"
    else:
        comment = "臉部與語音方向大致一致，可視為穩定的整體情緒。"

    return {
        "fusion_label": fusion_label,
        "is_conflict": conflict,
        "comment": comment,
    }


# ============================
# 工具：存暫存檔 + EXIF 旋轉修正
# ============================

def save_upload_to_tempfile(uploaded_file, suffix=".jpg") -> str:
    """把 streamlit 上傳物件存到暫存檔，回傳路徑"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def load_pil_from_upload(uploaded_file) -> Image.Image:
    """從上傳/拍照檔載入 PIL Image + 修正 EXIF 旋轉"""
    img_bytes = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)  # ✅ 修正手機照片常見旋轉問題
    return img


def save_pil_to_temp_jpg(img: Image.Image, quality=95) -> str:
    """把 PIL Image 存成暫存 jpg，回傳路徑"""
    img = img.convert("RGB")
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img.save(path, format="JPEG", quality=quality)
    return path


# ============================
# 工具：OpenCV 偵測最大人臉並裁切（手機整張照必備）
# ============================

def detect_and_crop_face(pil_img: Image.Image, margin_ratio=0.25):
    """
    回傳 (cropped_pil, bbox_or_none)
    bbox: (x0,y0,x1,y1) in original image coords
    """
    if not OPENCV_OK:
        return pil_img, None

    w, h = pil_img.size
    max_side = max(w, h)
    scale = 1.0
    pil_small = pil_img

    if max_side > 900:
        scale = 900.0 / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        pil_small = pil_img.resize((new_w, new_h))

    rgb = np.array(pil_small.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if faces is None or len(faces) == 0:
        return pil_img, None

    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    inv = 1.0 / scale
    x = int(x * inv); y = int(y * inv)
    fw = int(fw * inv); fh = int(fh * inv)

    pad = int(max(fw, fh) * margin_ratio)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + fw + pad)
    y1 = min(h, y + fh + pad)

    cropped = pil_img.crop((x0, y0, x1, y1))
    return cropped, (x0, y0, x1, y1)


# ============================
# ✅ 新增：把錄音 bytes 存成暫存 wav
# ============================

def save_audio_bytes_to_temp_wav(wav_bytes: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="Emotion Assistant Demo", layout="wide")

st.title(" Emotion Assistant Demo")
st.write("上傳/拍照臉部圖片＋語音＋文字，我會分析你的情緒並給出溫柔的安撫回應。")

# ======= 上傳區 =======
col_face, col_voice = st.columns(2)

with col_face:
    st.subheader("臉部圖片（手機拍照 / 相簿上傳）")
    cam_img = st.camera_input("用手機相機拍一張（建議正臉）")
    face_file = st.file_uploader("或從相簿上傳", type=["jpg", "jpeg", "png", "heic", "webp"])

    face_source = cam_img if cam_img is not None else face_file

    auto_crop = st.checkbox("自動偵測並裁切人臉（推薦：手機整張照）", value=True, disabled=not OPENCV_OK)
    margin_ratio = st.slider("裁切留白（margin）", 0.05, 0.50, 0.25, 0.05, disabled=not (OPENCV_OK and auto_crop))

    if not OPENCV_OK:
        st.info("（提示）你尚未安裝/載入 OpenCV，因此暫時不會自動裁臉。可 `pip install opencv-python`。")

with col_voice:
    st.subheader("語音（上傳 / 錄音二選一）")

    voice_mode = st.radio(
        "選擇語音來源",
        ["上傳 WAV", "直接錄音"],
        horizontal=True
    )

    voice_file = None
    recorded_wav_bytes = b""

    if voice_mode == "上傳 WAV":
        voice_file = st.file_uploader("上傳 (.wav)", type=["wav"], key="voice_uploader")
    else:
        st.markdown("按下開始→說話→停止，即可得到 WAV")
        audio_seg = audiorecorder("開始錄音", "停止錄音", key="voice_recorder")

        if len(audio_seg) > 0:
            recorded_wav_bytes = audio_seg.export(format="wav").read()
            st.audio(recorded_wav_bytes)
            st.caption(" 已錄到音：按『開始分析』會使用這段錄音。")

st.subheader(" 使用者文字內容")
user_text = st.text_area("你想說什麼？")

# ======= 分析按鈕 =======
if st.button("開始分析", use_container_width=True):

    # ---- 檢查必填 ----
    if face_source is None:
        st.error("請先提供臉部圖片（拍照或上傳）。")
        st.stop()

    if not user_text.strip():
        st.error("請輸入文字內容。")
        st.stop()

    if voice_mode == "上傳 WAV":
        if voice_file is None:
            st.error("你選了『上傳 WAV』，但尚未上傳檔案。")
            st.stop()
    else:
        if len(recorded_wav_bytes) == 0:
            st.error("你選了『直接錄音』，但尚未錄到音（請按開始/停止）。")
            st.stop()

    # ====== 臉部：讀取 + (可選) 裁臉 + 存成乾淨 jpg ======
    pil_img = load_pil_from_upload(face_source)

    bbox = None
    if auto_crop and OPENCV_OK:
        pil_face, bbox = detect_and_crop_face(pil_img, margin_ratio=margin_ratio)
    else:
        pil_face = pil_img

    face_path = save_pil_to_temp_jpg(pil_face, quality=95)

    # ====== 語音：依模式產生 voice_path ======
    if voice_mode == "直接錄音":
        voice_path = save_audio_bytes_to_temp_wav(recorded_wav_bytes)
    else:
        voice_path = save_upload_to_tempfile(voice_file, suffix=".wav")

    # ===== 情緒分析 =====
    face_info = analyze_face_for_gpt(face_path)
    voice_info = analyze_voice_for_gpt(voice_path)
    fusion = fusion_emotion(face_info, voice_info)

    # ===================================
    # 左右兩欄情緒展示區
    # ===================================
    left, right = st.columns(2)

    # ----- 左欄：臉部 -----
    with left:
        st.subheader(" 臉部情緒分析")

        st.markdown("**原始照片**")
        st.image(pil_img, caption="原始照片（手機通常是整張）", width=350)

        if bbox is not None:
            st.markdown("**裁切後的人臉**（送進模型的圖片）")
            st.image(pil_face, caption="Face Crop", width=250)
        else:
            st.markdown("**送進模型的圖片**（未裁臉）")
            st.image(pil_face, caption="Input to Model", width=250)

        st.write(f"主情緒：**{face_info['main_label']}**（信心度 {face_info['main_conf']:.2f}）")
        st.write("Top-3 分佈：")
        for lbl, prob in face_info["top3"]:
            st.write(f"- {lbl}: {prob:.2f}")

    # ----- 右欄：語音 -----
    with right:
        st.subheader("語音情緒分析")

        st.audio(voice_path)

        st.write(f"主情緒：**{voice_info['main_label']}**（信心度 {voice_info['main_conf']:.2f}）")
        st.write("Top-3 分佈：")
        for lbl, prob in voice_info["top3"]:
            st.write(f"- {lbl}: {prob:.2f}")

        # ===== 語音波形 & Mel-spectrogram =====
        try:
            y, sr = librosa.load(voice_path, sr=None)
            st.markdown("#####  語音波形 / Waveform")
            fig1, ax1 = plt.subplots(figsize=(6, 2))
            time_axis = np.linspace(0, len(y) / sr, num=len(y))
            ax1.plot(time_axis, y)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)

            st.markdown("##### Mel 頻譜圖 / Mel-Spectrogram")
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            img = librosa.display.specshow(
                S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax2
            )
            fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
            ax2.set_title("Mel-Spectrogram")
            st.pyplot(fig2)
            plt.close(fig2)

        except Exception as e:
            st.warning(f"語音視覺化失敗，但不影響情緒分類：{e}")

    # ===================================
    # Fusion 區域（置中）
    # ===================================
    st.subheader("多模態情緒整合 (Fusion)")

    if fusion["is_conflict"]:
        st.warning(
            f"整體情緒：**{fusion['fusion_label']}** （偵測到衝突）\n\n{fusion['comment']}"
        )
    else:
        st.success(
            f"整體情緒：**{fusion['fusion_label']}**\n\n{fusion['comment']}"
        )

    # ===================================
    # 準備 LLM Prompt
    # ===================================
    face_dist = ", ".join([f"{lbl}: {p:.2f}" for lbl, p in face_info["top3"]])
    voice_dist = ", ".join([f"{lbl}: {p:.2f}" for lbl, p in voice_info["top3"]])

    prompt = f"""
你是一個溫柔、有同理心的情緒輔助 AI。

=== 臉部情緒 ===
主情緒：{face_info['main_label']}（{face_info['main_conf']:.2f}）
分佈：{face_dist}

=== 語音情緒 ===
主情緒：{voice_info['main_label']}（{voice_info['main_conf']:.2f}）
分佈：{voice_dist}

=== 多模態整合 ===
整體情緒：{fusion['fusion_label']}
說明：{fusion['comment']}

=== 使用者輸入 ===
「{user_text}」

請生成：
1. 一句描述對方的情緒（可混合）。
2. 2～3 句溫柔安慰。
3. 一個具體的小建議。
"""

    reply = call_gpt_api(prompt)

    # ===================================
    # 寫入 log
    # ===================================
    logs_path = os.path.join(ROOT_DIR, "logs.csv")
    try:
        file_exists = os.path.exists(logs_path)
        with open(logs_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "face_main",
                        "face_conf",
                        "voice_main",
                        "voice_conf",
                        "fusion_label",
                        "user_text",
                        "ai_reply",
                    ]
                )
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    face_info["main_label"],
                    f"{face_info['main_conf']:.4f}",
                    voice_info["main_label"],
                    f"{voice_info['main_conf']:.4f}",
                    fusion["fusion_label"],
                    user_text.replace("\n", " "),
                    reply.replace("\n", " "),
                ]
            )
        st.caption(" 本次結果已記錄到 logs.csv，可用於後續統計與報告。")
    except Exception as e:
        st.warning(f"紀錄 log 時發生錯誤（不影響 Demo）：{e}")

    # ===================================
    # 顯示 AI 回應
    # ===================================
    st.subheader("AI 回應")
    st.write(reply)
