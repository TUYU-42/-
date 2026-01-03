import google.generativeai as genai

genai.configure(api_key="AIzaSyD3k5YCnZxnQwszL4RYpNThpyaJBLGqCVc")  # ← 換成你的金鑰

models = genai.list_models()
for m in models:
    print(m.name)
