#!/bin/bash
set -e

echo "=== 🧠 Installing Call-Center Mindset AI Stack on Ubuntu 22.04 ==="

# ---- 1️⃣ System Update & Base Packages ----
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip ffmpeg curl wget build-essential

# ---- 2️⃣ Create Project Directory ----
mkdir -p ~/callcenter_ai && cd ~/callcenter_ai

# ---- 3️⃣ Python Virtual Environment ----
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools

# ---- 4️⃣ Install Core Python Libraries ----
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets sentencepiece accelerate soundfile librosa numpy pandas scikit-learn tqdm

# ---- 5️⃣ Install Whisper.cpp (local speech-to-text) ----
echo "=== Installing Whisper.cpp ==="
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
# Download base model (or choose tiny/base/small/medium/large)
bash ./models/download-ggml-model.sh base
cd ..

# ---- 6️⃣ Install Emotion & Sentiment Models ----
mkdir -p models && cd models
# DistilBERT Emotion model
python3 - <<'PYCODE'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
models = [
    "bhadresh-savani/distilbert-base-uncased-emotion",
    "cardiffnlp/twitter-roberta-base-sentiment"
]
for m in models:
    print(f"Downloading {m} ...")
    tok = AutoTokenizer.from_pretrained(m)
    mod = AutoModelForSequenceClassification.from_pretrained(m)
    tok.save_pretrained(m.split('/')[-1])
    mod.save_pretrained(m.split('/')[-1])
PYCODE
cd ..

# ---- 7️⃣ Install Rasa NLU for Intent Classification ----
echo "=== Installing Rasa ==="
pip install rasa==3.6.18
rasa init --no-prompt --project rasa_project
# (You can later train intents using your own data in rasa_project/)

# ---- 8️⃣ Aggregator + Streamlit Dashboard ----
mkdir -p app && cd app
cat <<'PYAPP' > app.py
import streamlit as st, subprocess, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json

st.set_page_config(page_title="Call Center Emotion Dashboard", layout="wide")
st.title("🎧 Real-Time Call Emotion & Sentiment Analysis")

audio_file = st.file_uploader("Upload Call Audio (.wav or .mp3)", type=["wav","mp3"])
if audio_file:
    with open("temp_audio.wav","wb") as f: f.write(audio_file.read())
    st.info("Transcribing with Whisper.cpp...")
    result = subprocess.run(["./whisper.cpp/main","-m","whisper.cpp/models/ggml-base.bin","-f","temp_audio.wav","-otxt"],
                            capture_output=True, text=True)
    transcript = result.stdout if result.stdout else open("temp_audio.wav.txt").read()
    st.text_area("Transcript", transcript, height=200)

    # Load emotion & sentiment models
    emo_tok = AutoTokenizer.from_pretrained("../models/distilbert-base-uncased-emotion")
    emo_mod = AutoModelForSequenceClassification.from_pretrained("../models/distilbert-base-uncased-emotion")
    sen_tok = AutoTokenizer.from_pretrained("../models/twitter-roberta-base-sentiment")
    sen_mod = AutoModelForSequenceClassification.from_pretrained("../models/twitter-roberta-base-sentiment")

    # Run inference
    with torch.no_grad():
        emo_inputs = emo_tok(transcript, return_tensors="pt", truncation=True, padding=True)
        sen_inputs = sen_tok(transcript, return_tensors="pt", truncation=True, padding=True)
        emo_pred = torch.softmax(emo_mod(**emo_inputs).logits, dim=1)
        sen_pred = torch.softmax(sen_mod(**sen_inputs).logits, dim=1)
    emo_label = torch.argmax(emo_pred).item()
    sen_label = torch.argmax(sen_pred).item()

    st.subheader("🧠 Emotion Prediction")
    st.json(json.dumps({"label_id": int(emo_label), "probabilities": emo_pred.tolist()}))
    st.subheader("💬 Sentiment Prediction")
    st.json(json.dumps({"label_id": int(sen_label), "probabilities": sen_pred.tolist()}))
PYAPP
cd ..

# ---- 9️⃣ Install Streamlit & Run App ----
pip install streamlit

echo "=== ✅ Installation Complete ==="
echo "To start the app:"
echo "------------------------------------------------------------"
echo "cd ~/callcenter_ai && source venv/bin/activate"
echo "streamlit run app/app.py"
echo "------------------------------------------------------------"
echo "Whisper model: ~/callcenter_ai/whisper.cpp/models/ggml-base.bin"
echo "Emotion/Sentiment models: ~/callcenter_ai/models/"
echo "Rasa project: ~/callcenter_ai/rasa_project/"
echo "Dashboard: http://localhost:8501"
