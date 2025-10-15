# 🎧 Call Center Emotion & Sentiment AI (On-Prem Ubuntu 22.04)

An **open-source, on-premise AI pipeline** for analyzing **call center audio** to detect **emotion**, **sentiment**, and **intent** using local models — no cloud required.

---

## 🧠 Architecture Overview

```text
[Audio Input] 
   ↓
[Whisper.cpp - Speech to Text]
   ↓
[Transcript]
   ↓
[DistilBERT Emotion Model] + [RoBERTa Sentiment Model] + [Rasa Intent Classifier]
   ↓
[Aggregator (Python)]
   ↓
[Dashboard (Streamlit / Grafana)]

1️⃣ Download and Install
wget https://github.com/redhatmurali/Call-Center-Mindset/blob/main/install_callcenter_ai.sh
chmod +x install_callcenter_ai.sh
./install_callcenter_ai.sh


2️⃣ Train Rasa Intent Model
cd ~/callcenter_ai
source venv/bin/activate
streamlit run app/app.py
