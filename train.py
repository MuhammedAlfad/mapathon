import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ParkingDataset
from model import UNet

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

dataset = ParkingDataset(
    "dataset/images",
    "dataset/masks"
)


loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
weights = torch.tensor([0.4, 1.2, 1.0])  # obstacle, parkable, path
criterion = nn.CrossEntropyLoss(weight=weights)


for epoch in range(40):
    total_loss = 0
    for imgs, masks in loader:
        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "parking_model.pth")

