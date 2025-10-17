# 🎧 Call Center Emotion & Sentiment AI (On-Prem Ubuntu 22.04)

An **open-source, on-premise AI pipeline** for analyzing **call center audio** to detect **emotion**, **sentiment**, and **intent** using local models — no cloud required.

---

## 🧠 Architecture Overview

```text
[Audio Input] 
   ↓
[Whisper.cpp (or) WhisperX - Speech to Text]  large-v3
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

3️⃣ Run Streamlit Dashboard
cd ~/callcenter_ai
source venv/bin/activate
streamlit run app/app.py

4️⃣ (Optional) Set Up Grafana for Visualization
sudo apt install grafana -y
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

~/callcenter_ai/
├── venv/                     # Python virtual environment
├── whisper.cpp/              # Offline speech-to-text engine
│   └── models/ggml-base.bin
├── models/
│   ├── distilbert-base-uncased-emotion/
│   └── twitter-roberta-base-sentiment/
├── rasa_project/             # Intent detection project
├── app/
│   └── app.py                # Streamlit dashboard
└── install_callcenter_ai.sh  # Setup script



[Audio Input] 
   ↓
Wav2Vec2
   ↓
[DistilBERT Emotion Model] + [RoBERTa Sentiment Model] + [Rasa Intent Classifier]
   ↓
OUTPUT



