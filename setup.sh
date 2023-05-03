pip install -r requirements.txt
gdown "https://drive.google.com/uc?export=download&id=1-DLdSsCbvoDvdlCV2iE1NwrTARezZbKz" -O mask_check.pt
streamlit run mask_check.py --server.port 10000
