import torch
import torch.nn.functional as F

from data.datasets import AEFlowDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import AEFlow
from torchmetrics.functional import structural_similarity_index_measure as ssim
from utils import plotBatch, loss_function, CustomLogger
from datetime import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
root = '/home/manuel/ae-flow/src/data/chest_xray'
#root = "./data/chest_xray"
batch_size = 9
epochs = 10

dataset = AEFlowDataset(root=root, train=True,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
test_dataset = AEFlowDataset(root=root, train=False,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = AEFlow('fast_flow', device)
#model = AEFlow('res_net', device)
optimizer = torch.optim.Adam(model.parameters())
alpha = 0.5
beta = 0.9
global_step = 0
logger = CustomLogger(root + '/runs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))


for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for i, (x, _) in enumerate(dataloader):
        if i > 10:
            break
        optimizer.zero_grad()
        x = x.to(device).requires_grad_()
        y, log_prob, logdet_jac = model(x)
        #y, log_prob, logdet_jac = torch.utils.checkpoint.checkpoint(model, x)
        loss = loss_function(x, y, log_prob.mean(), logdet_jac.mean(), alpha = 1/2,SSIM = False)
        epoch_loss += loss.item()

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        global_step += 1
        anomaly_score = beta * (-torch.exp(log_prob).sum()) + (1 - beta) * ssim(y, x, reduction='sum')
        print(f"Train: epoch {epoch}, iteration: {global_step} train loss = {loss},")
        logger.log_all(global_step, loss, anomaly_score, x, y, None)
    if epoch % 2 == 0:
        model.eval()
        test_epoch_loss = 0
        epoch_anomaly_score = 0
        for i, (x, _) in enumerate(test_dataloader):
            if i > 10:
                break
            x = x.to(device)
            y, log_prob, logdet_jac = model(x)

            loss = loss_function(x, y, log_prob.mean(), logdet_jac.mean(), alpha = 1/2,SSIM = False)
            epoch_loss += loss.item()

            anomaly_score = beta * (-torch.exp(log_prob).sum()) + (1 - beta) * ssim(y, x, reduction='sum')
            epoch_anomaly_score += anomaly_score
            print(f"Test: epoch {epoch}, iteration: {global_step} train loss = {loss},")

        print(f"Test: epoch {epoch}, test loss = {test_epoch_loss/len(test_dataset)},"
              f" score  = {epoch_anomaly_score/len(test_dataset)}")
    print(f"Train: epoch {epoch}, test loss = {epoch_loss/len(dataset)}")

            