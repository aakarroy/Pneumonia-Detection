from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def precision_recall(y_true,y_prob):
    p,r,t = precision_recall_curve(y_true,y_prob)
    f1 = 2*(p*r)/(p+r+1e-10)
    best_idx = np.argmax(f1)
    best_t = t[best_idx]
    return best_t,f1[best_idx]

class PnuemoniaiTest(Dataset):
    def __init__(self,parent_folder) -> None:
        super().__init__()
        self.test_path = os.path.join(parent_folder,"Testing")
        self.normal_folder = os.path.join(self.test_path,"normal")
        self.pneumonia_folder = os.path.join(self.test_path,"pneumonia")
        self.images = []
        self.labels = []
        for img in os.listdir(self.normal_folder):
            self.images.append(os.path.join(self.normal_folder,img))
            self.labels.append(0)
        for img in os.listdir(self.pneumonia_folder):
            self.images.append(os.path.join(self.pneumonia_folder,img))
            self.labels.append(1)
        self.transform = Compose([
            Grayscale(1),
            ToTensor(),
            Resize((128,128))
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img_data = Image.open(img_path)
        return {"images":self.transform(img_data),"labels": self.labels[index]}

parent_folder = r"Dataset-of-Normal-and-Pneumo"
test_data = PnuemoniaiTest(parent_folder)
test_data_loader = DataLoader(test_data,batch_size=1,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
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
    nn.Flatten()
).to(device)

model.load_state_dict(torch.load(r"models\Best-Pnemonia-03.pth",map_location=device))

model.eval()

i = 1
correct = 0
total = 0
TP = 0
TN = 0
FP = 0
FN = 0
all_probs = []
all_labels = []
for batch_data in test_data_loader:
    images = batch_data["images"].to(device)
    labels = batch_data["labels"].to(device).view(-1)
    with torch.no_grad():
        raw_logits = model(images)
        prob = torch.sigmoid(raw_logits).view(-1).cpu().numpy()
        all_probs.extend(prob)
        all_labels.extend(labels.cpu().numpy())

best_t, f1= precision_recall(all_labels,all_probs)
print(f"Best Theshold is: {best_t:.4f} and best F1 score is: {f1:.4f}")
time.sleep(5)

for batch_data in test_data_loader:
    images = batch_data["images"].to(device)
    labels = batch_data["labels"].to(device).view(-1)
    with torch.no_grad():
        raw_logits = model(images)
        prob = torch.sigmoid(raw_logits).item()
        pred_class = 1 if prob>best_t else 0
        pred_confidence = prob if pred_class == 1 else (1.0-prob)
        print(f"Bath {i}")
        print(f"Prediction: {pred_class} Actual: {labels.item()} Confidence: {pred_confidence:.4f}")

        if(pred_class==labels==1):
            correct+=1
            TP+=1
        elif(pred_class==labels==0):
            correct+=1
            TN+=1
        elif(pred_class==1 and labels==0):
            FP+=1
        elif(pred_class==0 and labels==1):
            FN+=1
        
        total+=1
        i+=1
print(f"""
TP : {TP}
FP : {FP}
FN : {FN}
TN : {TN}
""")
plt.bar(x=["Total","TP","FP","FN","TN"],height=[total,TP,FP,FN,TN],color="red")
plt.savefig("Predictions-Metrics.png")
plt.show()
print(f"""Correct Prediction: {correct}/{total}
Accuracy: {(correct/total)*100:.2f}
""")

transform = Compose([
            Grayscale(1),
            ToTensor(),
            Resize((128,128))
])
test_normal_files = os.listdir(r"Dataset-of-Normal-and-Pneumo/Testing/normal")
test_pneumonia_files = os.listdir(r"Dataset-of-Normal-and-Pneumo/Testing/pneumonia")
rand_normal = random.sample(range(len(test_normal_files)),3)
rand_pneumonia = random.sample(range(len(test_pneumonia_files)),3)
fig, axis = plt.subplots(3,2,figsize=(10,15))
axis = axis.flatten()
all_data = [(os.path.join(r"Dataset-of-Normal-and-Pneumo/Testing/normal", test_normal_files[idx]),0) for idx in rand_normal] + \
            [(os.path.join(r"Dataset-of-Normal-and-Pneumo/Testing/pneumonia",test_pneumonia_files[idx]),1) for idx in rand_pneumonia]
for idx,ax in enumerate(axis):
    img_path,label = all_data[idx]
    img_data = Image.open(img_path)

    with torch.no_grad():
        processed_img = transform(img_data).unsqueeze(0).to(device)
        raw_logits = model(processed_img)
        prob = torch.sigmoid(raw_logits).item()
        
    pred_class = 1 if prob > best_t else 0
    confidence = prob if pred_class==1 else 1.0-prob

    ax.imshow(img_data)
    color = "green" if pred_class==label else "red"
    true_text = "Pneumonia" if label == 1 else "Normal"
    pred_text = "Pneumonia" if pred_class == 1 else "Normal"

    title = f"True: {true_text} Pred: {pred_text} Conf: {confidence*100:.2f}%"
    ax.set_title(title,color=color)
plt.tight_layout()
plt.savefig("Prediction.png")
plt.show()