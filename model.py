#%%
from tqdm import tqdm
import kagglehub
# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia") + "/chest_xray"
binMap = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}
print("Path to dataset files:", path)

#%%
import argparse
parser = argparse.ArgumentParser(prog="Model")
parser.add_argument(
    "-s",
    "--save",
    type=str, default=None)
args = parser.parse_args()


#%%
import os
train_normal = os.path.join(path, "train/NORMAL")
train_pneumonia = os.path.join(path, "train/PNEUMONIA")
print(f"images PNEUMONIA: {len(os.listdir(train_pneumonia))}")
print(f"images NORMAL: {len(os.listdir(train_normal))}")

#%% 
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(0.4),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.RandomRotation(25),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# %%
from torchvision import datasets    
from torchvision import transforms
train_dataset = datasets.ImageFolder(root=path + "/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=path + "/val", transform=test_transform)
test_dataset = datasets.ImageFolder(root=path + "/test", transform=test_transform)
print(f"Train len: {len(train_dataset)}")
print(f"Val len: {len(val_dataset)}")
print(f"Test len: {len(test_dataset)}\n\n")

# %%
import matplotlib.pyplot as plt
# plt.imshow(train_dataset[0][0].permute(1, 2, 0))
# plt.show()
# %%
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# %%
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 1),
        )
    def forward(self, x):
        return self.model(x)
# %%
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Model()
model.to(device)
model.train()
# %%
loss_per_epoch = []
accuracy = Accuracy(task="binary", num_classes=2).to(device)
precision = Precision(task="binary", num_classes=2).to(device)
recall = Recall(task="binary", num_classes=2).to(device)
f1_score = F1Score(task="binary", num_classes=2).to(device)
confusion_matrix = ConfusionMatrix(task="binary", num_classes=2).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2], device=device))
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(2):
    total_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1).float()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        probs = torch.sigmoid(outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        accuracy.update(probs, labels)
        precision.update(probs, labels)
        recall.update(probs, labels)
        f1_score.update(probs, labels)
        confusion_matrix.update(probs, labels)
    
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")
    loss_per_epoch.append([epoch, total_loss / len(train_loader)])
    print(f"Accuracy: {accuracy.compute()}")
    print(f"Precision: {precision.compute()}")
    print(f"Recall: {recall.compute()}")
    print(f"F1 Score: {f1_score.compute()}")
    print(f"Confusion Matrix:\n{confusion_matrix.compute()}")
    
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    confusion_matrix.reset()
    total_loss = 0
    loader = test_loader

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).float()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss
            probs = torch.sigmoid(outputs)
            accuracy.update(probs, labels)
            precision.update(probs, labels)
            recall.update(probs, labels)
            f1_score.update(probs, labels)
            confusion_matrix.update(probs, labels)

        loss_per_epoch[-1].append(total_loss / len(loader))
        print("\n____VALIDATION____")
        print(f"Accuracy: {accuracy.compute()}")
        print(f"Precision: {precision.compute()}")
        print(f"Recall: {recall.compute()}")
        print(f"F1 Score: {f1_score.compute()}")
        print(f"Confusion Matrix:\n{confusion_matrix.compute()}")
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()
        confusion_matrix.reset()
        print("\n\n\n")

plt.title("Chest XRAY Model Data")
epochs = [entry[0] for entry in loss_per_epoch]
train_losses = [
    i[1].cpu().item() if torch.is_tensor(i[1]) else i[1]
    for i in loss_per_epoch
]
val_losses = [
    i[2].cpu().item() if torch.is_tensor(i[2]) else i[2]
    for i in loss_per_epoch
]
plt.plot(epochs, train_losses, 'r-', label="Train")
plt.plot(epochs, val_losses, 'b-', label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")

if args.save:
    run_label = args.save
    run_dir = "Results"
    os.makedirs(run_dir, exist_ok=True)
    plot_path = os.path.join(run_dir, f"{run_label}.png")
    plt.savefig(plot_path)

plt.show()
