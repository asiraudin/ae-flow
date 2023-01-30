import torch
import torch.nn.functional as F

from data.datasets import AEFlowDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import AEFlow
from torchmetrics.functional import structural_similarity_index_measure as ssim


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

dataset = AEFlowDataset(root="./data/chest_xray", train=True,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
test_dataset = AEFlowDataset(root="./data/chest_xray", train=False,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

model = AEFlow()
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
alpha = 0.5
beta = 0.9

for epoch in range(3):
    model.train()
    epoch_loss = 0.0
    for i, (x, _) in enumerate(dataloader):
        if i > 10:
            break
        optimizer.zero_grad()
        x = x.to(device)
        y, log_prob, logdet_jac = model(x)

        loss = (1 - alpha) * F.mse_loss(x, y, reduction='sum') + alpha * (-log_prob - logdet_jac).sum()
        epoch_loss += loss.item()

        loss = loss.mean()
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        model.eval()
        test_epoch_loss = 0
        epoch_anomaly_score = 0
        for i, (x, _) in enumerate(test_dataloader):
            if i > 10:
                break
            x = x.to(device)
            y, log_prob, logdet_jac = model(x)

            loss = (1 - alpha) * F.mse_loss(x, y, reduction='sum') + alpha * (-log_prob - logdet_jac).sum()
            epoch_loss += loss.item()

            anomaly_score = beta * (-torch.exp(log_prob).sum()) + (1 - beta) * ssim(y, x, reduction='sum')
            epoch_anomaly_score += anomaly_score

        print(f"Test: epoch {epoch}, test loss = {test_epoch_loss/len(test_dataset)},"
              f" score  = {epoch_anomaly_score/len(test_dataset)}")
    print(f"Train: epoch {epoch}, test loss = {epoch_loss/len(dataset)}")
