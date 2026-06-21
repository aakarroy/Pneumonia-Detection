import streamlit as st
import time 
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(r"models/Best-Pnemonia-03.pth")
MAX_FILE = 10
ALLOWED_FILES = ["jpeg","png","jpg"]
IMG_SIZE = (128,128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    return nn.Sequential(
    nn.Conv2d(in_channels=1,
              out_channels=4,
              stride=2,
              kernel_size=3,
              bias=False),
    nn.BatchNorm2d(4),
    nn.Tanh(),

    nn.Conv2d(in_channels=4,
              out_channels=16,
              kernel_size=3,
              stride=2,
              bias=False),
    nn.BatchNorm2d(16),
    nn.Tanh(),

    nn.Conv2d(in_channels=16,
              out_channels=64,
              kernel_size=3,
              stride=2,
              bias=False),
    nn.BatchNorm2d(64),
    nn.Tanh(),

    nn.Conv2d(in_channels=64,
              out_channels=16,
              kernel_size=3,
              stride=2,
              bias=False),
    nn.BatchNorm2d(16),
    nn.Tanh(),

    nn.Conv2d(in_channels=16,
              out_channels=4,
              kernel_size=3,
              stride=2,
              bias=False),
    nn.BatchNorm2d(4),
    nn.Tanh(),

    nn.Conv2d(in_channels=4,
              out_channels=1,
              kernel_size=3,
              stride=2,
              bias=False),
    nn.BatchNorm2d(1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten())

@st.cache_resource(show_spinner=True)
def load_model(model_path, device):
    model = build_model().to(device)
    try:
        model.load_state_dict(torch.load(model_path,map_location=device))
    except Exception as exc:
        raise RuntimeError(f"Failed to load Model Weights: {exc}")
    model.eval()
    return model

transform = Compose([
    Grayscale(1),
    Resize(IMG_SIZE),
    ToTensor()
])

def get_prediction(upload_file):
    with st.status("Anlayzing X-Ray....") as status:
        st.write("PreProcessing Image....")
        uploaded_image = Image.open(upload_file)
        img_tensor = transform(uploaded_image).unsqueeze(0).to(DEVICE)
        time.sleep(1)
        st.write("Predicting Output...")
        model = load_model(MODEL_PATH,DEVICE)
        status.update(label="Analysing...",state="running")
        time.sleep(1)
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()
        pred_class = (1 if prob> 0.4 else 0)
        confidence = (prob if pred_class==1 else (1-prob))
        status.update(label="Analysis Complete",state="complete")
    _,c,_ = st.columns([1, 2, 1])
    with c:
        st.image(uploaded_image,caption="Uploaded X-ray Scan",width=300,)
        if(pred_class==1):
            st.error(f"Pneumonia Detected. \nConfidence = {confidence * 100:.2f}")
        else:
            st.success(f"No Pneumonia Detected. \nConfidence = {confidence * 100:.2f}%")
        time.sleep(1)

def main():
    st.title("AI powered Pneumonia Detection",anchor=False,text_alignment="center")
    with st.spinner("Initializing AI Model...."):
        if(not MODEL_PATH.exists()):
                st.error("Model is not found update model path")
                st.stop()
    "---"
    _,c,_ = st.columns([1,50,1])
    with c:
        st.info("Upload your chest X-ray scans (JPG/PNG/JPEG). Our AI model provides an instant analysis.",width="stretch")
        
    upload_file = st.file_uploader("Upload your X-Ray",accept_multiple_files=False,type=ALLOWED_FILES) 
    if(upload_file is not None):
        file_size = upload_file.size / (1024*1024)
        if(file_size>MAX_FILE):
            st.write(f"Current File is {file_size}MB.\nMax upload Limit is {MAX_FILE} upload again.")
        else:
            with st.spinner("File is uploading...."):
                time.sleep(2)
            st.success("File upload successfull.")
            get_prediction(upload_file)
    
    "---"
    st.caption("""⚠️ This tool is a research/educational demo and not a certified medical device. \nIt must not be used as a substitute for professional diagnosis. Always consult a
licensed radiologist or physician.""")

if __name__=="__main__":
    main()

