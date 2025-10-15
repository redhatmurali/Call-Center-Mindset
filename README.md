# Call-Center-Mindset
Ubuntu 22.04 installation script that sets up your call-center emotion-analysis pipeline:

[Audio Input] 
 → [Whisper.cpp] 
 → [Transcript] 
 → [DistilBERT Emotion Model + RoBERTa Sentiment + Rasa Intent]
 → [Aggregator (Python)]
 → [Dashboard (Streamlit/Grafana)]


 wget https://your-domain-or-github/install_callcenter_ai.sh
chmod +x install_callcenter_ai.sh
./install_callcenter_ai.sh


cd ~/callcenter_ai/rasa_project
rasa train
rasa shell nlu

sudo apt install grafana -y
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
