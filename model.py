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
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
print(f"Test len: {len(test_dataset)}")

# %%
import matplotlib.pyplot as plt
plt.imshow(train_dataset[0][0].permute(1, 2, 0))
plt.show()
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
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(4 * 28 * 28, 1),
        )
    def forward(self, x):
        return self.model(x)
# %%
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Model()
model.to(device)
model.train()
# %%
accuracy = Accuracy(task="binary", num_classes=2).to(device)
precision = Precision(task="binary", num_classes=2).to(device)
recall = Recall(task="binary", num_classes=2).to(device)
f1_score = F1Score(task="binary", num_classes=2).to(device)
confusion_matrix = ConfusionMatrix(task="binary", num_classes=2).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(15):
    total_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(train_loader)):
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

    for i, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).float()
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        accuracy.update(probs, labels)
        precision.update(probs, labels)
        recall.update(probs, labels)
        f1_score.update(probs, labels)
        confusion_matrix.update(probs, labels)

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
