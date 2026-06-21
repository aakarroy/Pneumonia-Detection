from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, RandomAffine, RandomRotation
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from PIL import Image

class PneumoniaTrainData(Dataset):
    def __init__(self,parent_folder) -> None:
        super().__init__()
        train_path = os.path.join(parent_folder,"Training")
        # normal = 0, pneumonia = 1
        normal_folder = os.path.join(train_path,"normal")
        pneumonia_folder = os.path.join(train_path,"pneumonia")

        self.train_size = int(len(os.listdir(normal_folder))*0.8)
        self.images = []
        self.labels = []
        i = 0
        for img in os.listdir(normal_folder):
            if(i==self.train_size):
                break
            self.images.append(os.path.join(normal_folder,img))
            self.labels.append(0)
            i+=1
        i = 0
        for img in os.listdir(pneumonia_folder):
            if(i==self.train_size):
                break
            self.images.append(os.path.join(pneumonia_folder,img))
            self.labels.append(1)
        self.data = {"images":self.images,"labels":self.labels}
        self.transform  = Compose([
            Grayscale(1),
            RandomRotation(degrees=10),
            RandomAffine(translate=(0.05,0.05),degrees=0),
            Resize((128,128)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        image_data = Image.open(img_path)
        return {"images":self.transform(image_data),"labels":self.labels[index]}

class PneumoniaValData(Dataset):
    def __init__(self,parent_folder) -> None:
        super().__init__()
        val_path = os.path.join(parent_folder,"Training")
        self.images = []
        self.labels = []
        i = 0
        normal_folder = os.path.join(val_path,"normal")
        pneumonia_folder = os.path.join(val_path,"pneumonia")
        self.train_size = len(os.listdir(normal_folder))*0.8
        for img in os.listdir(normal_folder):
            if(i>self.train_size):
                self.images.append(os.path.join(normal_folder,img))
                self.labels.append(0)
            i+=1
        i= 0
        for img in os.listdir(pneumonia_folder):
            if(i>self.train_size):
                self.images.append(os.path.join(pneumonia_folder,img))
                self.labels.append(1)
            i+=1
        self.data = {"images": self.images,"labels":self.labels}
        self.transform = Compose([
            Grayscale(1),
            Resize((128,128)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image_data = Image.open(img_path)
        return {"images":self.transform(image_data),"labels":self.labels[index]}

    
parent_folder = r"Pneumonia-Detection\Dataset-of-Normal-and-Pneumo"
train_data = PneumoniaTrainData(parent_folder)
val_data = PneumoniaValData(parent_folder)
train_data_loader = DataLoader(train_data,batch_size=8,shuffle=True)
val_data_loader = DataLoader(val_data,batch_size=8,shuffle=True)

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

optim = Adam(model.parameters(),lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optim)
epochs = 30
loss_func = nn.BCEWithLogitsLoss()
best_val_loss = float("inf")


for epoch in range(1,epochs+1):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_data_loader,desc=f"Epoch {epoch}/{epochs} Train: ")
    train_correct = 0
    train_total = 0
    for batch_data in train_bar:
        images = batch_data["images"].to(device)
        labels = batch_data["labels"].to(device).float().view(-1,1)
        optim.zero_grad()
        predictions = model(images)
        loss = loss_func(predictions,labels)
        loss.backward()
        optim.step()
        train_loss +=loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(predictions)
            predicted_classes = (probs >0.8).float()
            train_correct +=(predicted_classes==labels).sum().item()
            train_total += labels.shape[0]
            acc = (train_correct/train_total)*100

        train_bar.set_postfix(loss=f"{loss.item():.4f}",acc=f"{acc:.2f}")

    avg_train_loss = train_loss/len(train_data_loader)
    print(f"{epoch}: Train Loss -> {avg_train_loss}")
    
    if(epoch%2==0):
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_data_loader,desc=f"Epoch {epoch} Validation: ")
        val_correct = 0
        val_total = 0
        val_acc = 0
        with torch.no_grad():
            for batch_data in val_bar:
                images = batch_data["images"].to(device)
                labels = batch_data["labels"].to(device).float().view(-1,1)
                predictions = model(images)
                loss = loss_func(predictions,labels)
                val_loss+=loss.item()
                probs = torch.sigmoid(predictions)
                predicted_classes = (probs>0.8).float()
                val_correct += (predicted_classes==labels).sum().item()
                val_total +=labels.shape[0]
                acc = (val_correct/val_total)*100                
                if(acc>val_acc):
                    val_acc = acc
                val_bar.set_postfix(loss=f"{loss.item():.4f}",acc=f"{acc:.2f}")
        avg_val_loss = val_loss/(len(val_data_loader))
        scheduler.step(avg_val_loss)
        if(avg_val_loss<best_val_loss):
            torch.save(model.state_dict(),r"Pneumonia-Detection\Best-Pnemonia-03.pth")
            best_val_loss = avg_val_loss
            print("Model Saved Sucessfully")

print(f"""Model Training succesfull for {epochs}
Train Loss: {avg_train_loss}
Best Val Accuracy {val_acc}""")


