
Emotion-Aware Assistant  
Facial and Voice Emotion Recognition System

---

1. 專案簡介

本專案為一套情緒感知系統（Emotion-Aware Assistant），  
透過臉部影像與語音訊號進行情緒辨識，使系統能夠理解使用者的情緒狀態，  
並作為後續人工智慧互動系統（如聊天機器人或輔助系統）的決策依據。


---

 2. 專案動機

在現代人機互動中，單純的文字或指令輸入已無法完整反映使用者的狀態。  
情緒是影響互動品質的重要因素，因此本專案的目標為：

- 讓系統能自動辨識使用者情緒
- 提升互動系統的智慧化與人性化
- 建立可擴充之多模態情緒分析架構

---

 3. 系統架構

系統整體流程如下：

1. 使用者提供臉部影像與語音輸入  
2. 臉部影像送入臉部情緒辨識模型  
3. 語音訊號送入語音情緒辨識模型  
4. 系統輸出各模組之情緒分類結果  
5. 情緒結果可供後續生成式 AI 使用或紀錄分析  

本架構採用模組化設計，各功能模組彼此獨立，方便維護與擴充。

---

 4. 功能說明

4.1 臉部情緒辨識
- 輸入格式：影像檔（jpg / png）
- 功能：分析臉部表情特徵
- 輸出：情緒分類結果（如 Happy、Sad、Angry 等）

4.2 語音情緒辨識
- 輸入格式：語音檔（wav）
- 功能：擷取語音特徵並進行情緒分類
- 輸出：情緒分類結果

 4.3 情緒紀錄
- 每次辨識結果會紀錄於 CSV 檔案
- 包含時間與各模組之情緒結果
- 可用於後續分析或模型改進

---

 5. 使用技術（Technologies）

- Python
- TensorFlow / Keras（深度學習模型）
- OpenCV（影像處理）
- Librosa（語音特徵擷取）
- NumPy、Pandas
- 生成式 AI 架構設計（可擴充 GPT 或 Gemini）

---

 6. 傳統 AI 與生成式 AI 結合說明

6.1 傳統 AI 部分

- 臉部情緒辨識屬於影像分類問題
- 語音情緒辨識屬於語音特徵分析與分類問題
- 兩者皆為監督式學習模型

6.2 生成式 AI 結合方式

- 傳統 AI 所輸出的情緒結果可作為生成式語言模型的輸入條件
- 生成式 AI 可依據情緒狀態調整回應內容與語氣
- 本專案已完成系統架構設計，具備實際整合可行性

---

7. 專案結構（Project Structure）


.
├─ app/
│  ├─ main.py
│  └─ app_ui.py
│
├─ face_emotion/
│  └─ predict_face.py
│
├─ voice_emotion/
│  └─ predict_voice.py
│
├─ logs.csv
├─ temp_face.jpg
├─ temp_voice.wav
├─ requirements.txt
└─ README.md



---

 8. 安裝與執行方式（Installation and Execution）

8.1 環境需求
- Python 3.9~3.12(for tesorflow)

8.2 安裝步驟

```bash
git clone https://github.com/TUYU-42/-
cd -
pip install -r requirements.txt
````

8.3 執行方式

```bash
python app/main.py
```

或使用圖形介面：

```bash
streamlit run app/app_ui.py
```

---

 9. 實驗結果與輸出

* 系統能正常輸出臉部與語音情緒分類結果
* 每次執行結果皆會紀錄於 logs.csv
* 可作為後續資料分析或系統優化之依據

---


 主題新穎性與實用性

本專案以情緒感知系統為主題，屬於 AI 與人機互動領域之重要應用方向，
具備實際應用價值與延伸性。

 傳統 AI 與生成式 AI 結合

本系統以傳統 AI 負責情緒分析，並設計可與生成式 AI 整合之架構，
符合課程專題之技術要求。

程式品質

* 程式可正確執行
* 功能完整


---

團隊分工

1123305 33.3% 程式開發
1121427 33.3% 模型改良
1121419 33.3% 程式骨架 


---
