
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
7. 專案架構設計（Detailed System Architecture）

本專案採用「多模態情緒分析＋生成式 AI 回應」之架構，  
整體系統可分為五個主要層級：
1. 使用者互動層（UI / Streamlit）
2. 臉部情緒辨識模組（Face Emotion Module）
3. 語音情緒辨識模組（Voice Emotion Module）
4. 多模態情緒融合（Fusion Logic）
5. 生成式 AI 回應與紀錄（LLM & Logging）
各層級彼此獨立，僅透過明確的介面傳遞資料，確保系統模組化與可維護性。
---

7.1 使用者互動層（UI Layer）

對應檔案：
- `app/app_ui.py`

此層負責所有使用者互動與流程控制，包含：

- 臉部圖片取得（手機拍照或檔案上傳）
- 語音輸入（上傳 WAV 或即時錄音）
- 使用者文字輸入
- 呼叫後端情緒分析與生成式 AI
- 顯示分析結果與視覺化內容

主要流程：
1. 由 Streamlit 介面收集三種輸入：
   - 臉部影像
   - 語音
   - 文字內容
2. 將影像與語音存為暫存檔
3. 呼叫臉部與語音情緒分析模組
4. 進行多模態情緒融合
5. 組合 Prompt 並呼叫生成式 AI
6. 顯示結果並寫入紀錄檔

此層不包含任何模型細節，僅負責流程協調。

---
7.2 臉部情緒辨識模組（Face Emotion Module）

對應檔案：
- `face_emotion/predict_face.py`
- `face_emotion/train_face_model.py`

(1) 前處理流程

在推論階段，臉部影像會被轉換為模型可接受的格式：

- 灰階影像
- Resize 至 48 × 48
- 正規化至 [0, 1]
- Shape 為 (1, 48, 48, 1)

此設計與訓練階段完全一致，避免 training–inference mismatch。
 (2) 模型設計

臉部情緒模型為 CNN 架構，採用 ResNet-like 殘差設計：
- 多層卷積與 Batch Normalization
- 殘差連接（Residual Blocks）
- Global Average Pooling
- Dense + Dropout 防止 overfitting
- Softmax 輸出 7 類情緒：
  angry, disgust, fear, happy, neutral, sad, surprise

(3) 推論prompt

```
analyze_face_for_gpt(image_path) -> dict
```

回傳內容包含：
- 主情緒標籤
- 主情緒信心度
- Top-3 情緒機率分佈
此回傳格式為後續多模態融合與生成式 AI 使用的標準化介面。
---
7.3 語音情緒辨識模組（Voice Emotion Module）

對應檔案：
- `voice_emotion/predict_voice.py`
- `voice_emotion/train_voice_model.py`
- `voice_emotion/organize_crema.py`

(1) 音訊前處理

所有語音輸入皆統一為：

- 單聲道
- 取樣率 16 kHz
- 固定長度 3 秒
- 去除前後長段靜音
- 音量正規化

在推論階段使用「中間裁切」，確保穩定性。

 (2) 特徵擷取

語音特徵為固定長度向量（248 維），包含：

- MFCC（40 維）及其一階、二階差分
- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- 所有特徵皆取 mean 與 standard deviation

 (3) 模型設計

- 使用 SVM（RBF kernel）
- Pipeline 結合 StandardScaler
- GridSearchCV 搜尋最佳超參數
- 使用 balanced class weight 處理類別不平衡

(4) 推論與不確定性判斷

推論結果包含：
- 主情緒
- 信心度
- Top-3 機率分佈
- margin（Top-1 與 Top-2 差距）
- 不確定性判斷（低信心或小 margin）

此資訊會被後續的多模態融合邏輯使用。
---

7.4 多模態情緒融合（Fusion Logic）

對應位置：
- `app/app_ui.py` 中的 `fusion_emotion` 函式

融合策略如下：

1. 若語音模型判定為不確定，則以臉部情緒為主
2. 否則比較臉部與語音信心度，取信心較高者
3. 將情緒分為 positive / negative / neutral 群組
4. 若臉部與語音方向相反，標記為情緒衝突
5. 產生文字說明，描述整體情緒狀態

此設計使系統能處理「表情與聲音不一致」的情境。

---

7.5 生成式 AI 回應與紀錄（LLM & Logging）

對應檔案：
- `app/main.py`
- `list_models.py`
(1) Prompt 組合設計

系統會將以下資訊整合為 Prompt：
- 臉部主情緒與分佈
- 語音主情緒與分佈
- 多模態融合結果與說明
- 使用者原始文字輸入

Prompt 明確引導生成式 AI：
- 描述使用者情緒
- 提供安撫回應
- 給出具體建議

(2) 生成式 AI 呼叫

- 使用 Google Gemini API
- 若 API 呼叫失敗，系統會回傳本地 fallback 回應
- 確保 Demo 與評分時系統穩定執行

(3) 紀錄機制

每次分析結果皆會寫入 `logs.csv`，內容包含：
- 時間戳記
- 臉部與語音主情緒
- 融合後情緒
- 使用者文字
- 生成式 AI 回應

此資料可用於後續統計分析與報告。

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
