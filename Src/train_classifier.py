import os
import torch
import cv2 
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from classifier import SignClassifier
from sklearn.model_selection import train_test_split

class SignDataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_path, transform=None):
        self.df = df
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.df.iloc[idx]['Path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[idx]['ClassId']
        if self.transform:
            image = self.transform(image)
        return image, label

def train(dataset_path, epochs=50, batch_size=32, output='classifier.pth', patience=5):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((64, 64)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(os.path.join(dataset_path, "Train.csv"))
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['ClassId'])

    train_loader = DataLoader(SignDataset(train_df, dataset_path, train_transform),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(SignDataset(val_df, dataset_path, val_transform),
                            batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader.dataset):.4f} '
              f'| Val Loss: {val_loss/len(val_loader.dataset):.4f} | Val Acc: {acc:.4f}')

   
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    print(f'Training complete. Best model saved to {output}')










'''
import os
import torch
import cv2 
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from classifier import SignClassifier
from sklearn.model_selection import train_test_split

class SignDataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_path, transform=None):
        self.df = df
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.df.iloc[idx]['Path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        label = self.df.iloc[idx]['ClassId']
        return self.transform(image), label

def train(dataset_path, epochs=30, batch_size=32, output='classifier.pth'):
    # Define proper transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(os.path.join(dataset_path, "Train.csv"))
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['ClassId'])
    
    train_loader = DataLoader(
        SignDataset(train_df, dataset_path, train_transform),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        SignDataset(val_df, dataset_path, val_transform),
        batch_size=batch_size
    )
    
    model = SignClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    best_acc = 0
    for epoch in range(epochs):
        # Training loop
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output)
        
        print(f'Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}')
        
    print(f'Training complete. Best model saved to {output}')

    '''