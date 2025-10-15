# ==========================================================
# 🧠 Call Center Mindset AI Stack Installer for Windows (PS 5)
# ==========================================================

Write-Host "=== Installing Call-Center Mindset AI Stack (Windows, PowerShell 5) ==="

# ---- 1️⃣ Pre-Check ----
if (-not (Test-Path "C:\Program Files\Python311\python.exe")) {
    Write-Host "Please install Python 3.10+ first from https://www.python.org/downloads/windows/" -ForegroundColor Yellow
    Write-Host "Then re-run this script."
    exit
}

# ---- 2️⃣ Create Project Directory ----
$ProjectPath = "$env:USERPROFILE\callcenter_ai"
if (-not (Test-Path $ProjectPath)) { New-Item -ItemType Directory -Path $ProjectPath | Out-Null }
Set-Location $ProjectPath

# ---- 3️⃣ Create Virtual Environment ----
& "C:\Program Files\Python311\python.exe" -m venv venv
cmd /c "venv\Scripts\activate && python -m pip install --upgrade pip wheel setuptools"

# ---- 4️⃣ Install Core Python Libraries ----
cmd /c "venv\Scripts\activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
cmd /c "venv\Scripts\activate && pip install transformers datasets sentencepiece accelerate soundfile librosa numpy pandas scikit-learn tqdm streamlit rasa==3.6.18"

# ---- 5️⃣ Install Whisper.cpp (Speech-to-Text) ----
Write-Host "Downloading Whisper.cpp ..."
if (-not (Test-Path "$ProjectPath\whisper.cpp")) { git clone https://github.com/ggerganov/whisper.cpp.git }
Set-Location "$ProjectPath\whisper.cpp"
Write-Host "Downloading base model ..."
Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -OutFile ".\ggml-base.bin"
Set-Location $ProjectPath

# ---- 6️⃣ Download Emotion & Sentiment Models ----
Write-Host "Downloading Hugging Face models (this may take a few minutes) ..."
cmd /c "venv\Scripts\activate && python - <<END
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
os.makedirs('models', exist_ok=True)
for m in ['bhadresh-savani/distilbert-base-uncased-emotion','cardiffnlp/twitter-roberta-base-sentiment']:
    print(f'Downloading {m} ...')
    tok = AutoTokenizer.from_pretrained(m)
    mod = AutoModelForSequenceClassification.from_pretrained(m)
    save = m.split('/')[-1]
    tok.save_pretrained(os.path.join('models', save))
    mod.save_pretrained(os.path.join('models', save))
END"

# ---- 7️⃣ Create Streamlit Dashboard App ----
$AppPath = "$ProjectPath\app"
if (-not (Test-Path $AppPath)) { New-Item -ItemType Directory -Path $AppPath | Out-Null }

@'
import streamlit as st, subprocess, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json

st.set_page_config(page_title="Call Center Emotion Dashboard", layout="wide")
st.title("🎧 Real-Time Call Emotion & Sentiment Analysis")

audio_file = st.file_uploader("Upload Call Audio (.wav or .mp3)", type=["wav","mp3"])
if audio_file:
    with open("temp_audio.wav","wb") as f: f.write(audio_file.read())
    st.info("Transcribing with Whisper.cpp (expect ~1min for 30s audio)...")
    result = subprocess.run(["..\\whisper.cpp\\main.exe","-m","..\\whisper.cpp\\ggml-base.bin","-f","temp_audio.wav","-otxt"],
                            capture_output=True, text=True)
    transcript = result.stdout if result.stdout else open("temp_audio.wav.txt").read()
    st.text_area("Transcript", transcript, height=200)

    emo_tok = AutoTokenizer.from_pretrained("../models/distilbert-base-uncased-emotion")
    emo_mod = AutoModelForSequenceClassification.from_pretrained("../models/distilbert-base-uncased-emotion")
    sen_tok = AutoTokenizer.from_pretrained("../models/twitter-roberta-base-sentiment")
    sen_mod = AutoModelForSequenceClassification.from_pretrained("../models/twitter-roberta-base-sentiment")

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
'@ | Out-File "$AppPath\app.py" -Encoding UTF8

Write-Host ""
Write-Host "=== ✅ Installation Complete ==="
Write-Host ""
Write-Host "To start the app:"
Write-Host "------------------------------------------------------------"
Write-Host "cd $ProjectPath"
Write-Host "venv\Scripts\activate"
Write-Host "streamlit run app\app.py"
Write-Host "------------------------------------------------------------"
Write-Host "Then open: http://localhost:8501"
