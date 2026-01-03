# main.py － Emotion Assistant Demo（Gemini 版）

import os
import sys
import textwrap

# === 讓 Python 找得到 face_emotion / voice_emotion 模組 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "face_emotion"))
sys.path.append(os.path.join(BASE_DIR, "voice_emotion"))

from predict_face import analyze_face_for_gpt
from predict_voice import analyze_voice_for_gpt

# === Gemini (Google AI Studio) ===
from google import genai


# 從環境變數讀取 API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("請先在系統環境變數裡設定 GEMINI_API_KEY")

# 建立新版 Client
client = genai.Client(api_key=GEMINI_API_KEY)

# 可以改成你想用的模型名稱（建議用 AI Studio 上顯示的）
MODEL_NAME = "gemini-2.5-flash"   # 或 "gemini-2.5-flash" 等等


def call_gpt_api(prompt: str) -> str:
    """
    使用新版 Gemini Client API 呼叫聊天模型。
    如果 API 沒有回文字，就回傳一段本地 fallback 文。
    """
    fallback_reply = (
        "我感覺到現在的情緒有一點複雜，"
        "好像混著一些疲累和壓力在裡面……\n"
        "你真的已經很努力了，先給自己一點時間好好休息，"
        "哪怕只是深呼吸幾次、喝點水，也是在好好照顧自己喔。"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        # 新版 API 直接用 response.text 取文字
        text = getattr(response, "text", None)
        if not text:
            return fallback_reply

        return text.strip()

    except Exception as e:
        print("[ERROR] Gemini 呼叫失敗:", e)
        return fallback_reply


def build_prompt_for_gpt(user_text: str, face_info: dict, voice_info: dict) -> str:
    """
    把臉部情緒 + 語音情緒 + 文字內容整合成一個 prompt。
    """
    face_main = face_info["main_label"]
    face_conf = face_info["main_conf"]
    face_top3 = face_info["top3"]

    voice_main = voice_info["main_label"]
    voice_conf = voice_info["main_conf"]
    voice_top3 = voice_info["top3"]

    def format_top3(top3):
        lines = []
        for label, prob in top3:
            lines.append(f"- {label}: {prob:.2f}")
        return "\n".join(lines)

    prompt = f"""
    你是一個溫柔、有同理心的情緒輔助聊天 AI。

    請根據「臉部情緒」＋「語音情緒」＋「使用者文字」來回應，
    重點是安撫、理解對方，並給一個具體的小建議。

    === 臉部情緒分析 ===
    臉部主情緒：{face_main}（信心度 {face_conf:.2f}）
    臉部情緒分布（Top-3）：
    {format_top3(face_top3)}

    === 語音情緒分析 ===
    語音主情緒：{voice_main}（信心度 {voice_conf:.2f}）
    語音情緒分布（Top-3）：
    {format_top3(voice_top3)}

    === 使用者說的話 ===
    \"\"\"{user_text}\"\"\"

    請輸出：
    1. 先簡短描述你感覺到的情緒狀態（可以是混合情緒）。
    2. 再給一段 2～3 句溫柔的回應與建議。
    請用口語化、溫柔的語氣，字數不要太長。
    """
    return textwrap.dedent(prompt).strip()


def main():
    print("=== Emotion Assistant Demo ===")

    face_path = input("請輸入臉部圖片路徑（例如 ../face_emotion/test.jpg）: ").strip()
    voice_path = input("請輸入語音檔路徑（例如 ../voice_emotion/test.wav）: ").strip()
    user_text = input("請輸入使用者說的內容: ").strip()

    # 1. 臉部情緒分析
    face_info = analyze_face_for_gpt(face_path)
    print("\n[臉部情緒分析]")
    print(f"臉部主情緒：{face_info['main_label']}（信心度 {face_info['main_conf']:.2f}）")
    print("臉部情緒分布（Top-3）：")
    for label, prob in face_info["top3"]:
        print(f"- {label}: {prob:.2f}")

    # 2. 語音情緒分析
    voice_info = analyze_voice_for_gpt(voice_path)
    print("\n[語音情緒分析]")
    print(f"語音主情緒：{voice_info['main_label']}（信心度 {voice_info['main_conf']:.2f}）")
    print("語音情緒分布（Top-3）：")
    for label, prob in voice_info["top3"]:
        print(f"- {label}: {prob:.2f}")

    # 3. 建立給 Gemini 的 prompt（也印出來當作報告用）
    prompt = build_prompt_for_gpt(user_text, face_info, voice_info)
    print("\n[送給 GPT / Gemini 的 Prompt 範例]")
    print("--------------------------------")
    print(prompt)
    print("--------------------------------")

    # 4. 呼叫 Gemini 產生回覆
    reply = call_gpt_api(prompt)

    print("\n[GPT / Gemini 回覆]")
    print(reply)


if __name__ == "__main__":
    main()
 