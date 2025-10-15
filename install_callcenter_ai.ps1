<#
Title: Call Center Mindset AI Stack - Windows Installation Script
Author: ChatGPT GPT-5
Platform: Windows 10/11 (PowerShell 7+)
#>

Write-Host "=== 🧠 Installing Call-Center Mindset AI Stack on Windows ===" -ForegroundColor Cyan

# ---- 1️⃣ Check for Admin Privileges ----
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "Please run PowerShell as Administrator!" -ForegroundColor Red
    exit
}

# ---- 2️⃣ Ensure Required Packages ----
Write-Host "Installing dependencies..."
winget install -e --id Git.Git -h
winget install -e --id Python.Python.3.11 -h
winget install -e --id FFmpeg.FFmpeg -h

# Refresh environment so Python is visible
$env:Path += ";C:\Program Files\Python311\Scripts;C:\Program Files\Python311\"

# ---- 3️⃣ Project Directory ----
New-Item -ItemType Directory -Path "$env:USERPROFILE\callcenter_ai" -Force | Out-Null
Set-Location "$env:USERPROFILE\callcenter_ai"

# ---- 4️⃣ Python Virtual Environment ----
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools

# ---- 5️⃣ Install Core Python Libraries ----
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets sentencepiece accelerate soundfile librosa numpy pandas scikit-learn tqdm

# ---- 6️⃣ Install Whisper.cpp (Windows) ----
Write-Host "=== Installing Whisper.cpp ==="
git clone https://github.com/ggerganov/whisper.cpp.git
Set-Location whisper.cpp
# Build (requires MSBuild / Visual Studio Build Tools)
if (Test-Path "msvc") {
    Write-Host "Building whisper.cpp using MSVC..."
    msbuild .\examples\main\main.vcxproj /p:Configuration=Release
} else {
    Write-Host "Please ensure Visual Studio Build Tools or CMake is installed."
}
# Download base model
Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -OutFile ".\models\ggml-base.bin"
Set-Location ..

# ---- 7️⃣ Download Emotion & Sentiment Models ----
New-Item -ItemType Directory -Path ".\models" -Force | Out-Null
python - <<'PYCODE'
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

# ---- 8️⃣ Install Rasa ----
Write-Host "=== Installing Rasa ==="
pip install rasa==3.6.18
rasa init --no-prompt --project rasa_project

# ---- 9️⃣ Create Streamlit App ----
New-Item -ItemType Directory -Path ".\app" -Force | Out-Null
Set-Location .\app
@"
import streamlit as st, subprocess, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json

st.set_page_config(page_title="Call Center Emotion Dashboard", layout="wide")
st.title("🎧 Real-Time Call Emotion & Sentiment Analysis")

audio_file = st.file_uploader("Upload Call Audio (.wav or .mp3)", type=["wav","mp3"])
if audio_file:
    with open("temp_audio.wav","wb") as f: f.write(audio_file.read())
    st.info("Transcribing with Whisper.cpp...")
    result = subprocess.run(["..\\whisper.cpp\\Release\\main.exe","-m","..\\whisper.cpp\\models\\ggml-base.bin","-f","temp_audio.wav","-otxt"],
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
"@ | Out-File -Encoding utf8 app.py
Set-Location ..

# ---- 🔟 Install Streamlit ----
pip install streamlit

Write-Host "=== ✅ Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start the app:" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------"
Write-Host "cd `$env:USERPROFILE\callcenter_ai"
Write-Host "venv\Scripts\Activate.ps1"
Write-Host "streamlit run app\app.py"
Write-Host "------------------------------------------------------------"
Write-Host "Dashboard: http://localhost:8501"
