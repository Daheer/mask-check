pip install -r requirements.txt
gdown "https://drive.google.com/uc?export=download&id=1-HSL744Kbxa3Fc-YZNQPQHFHtxcq340W" -O mask_check.pt
streamlit run mask_check.py --server.port 10000
