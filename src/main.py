import torch
import torch.nn.functional as F
import numpy as np

from data.datasets import AEFlowDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import AEFlow
from torchmetrics.functional import structural_similarity_index_measure as ssim
from utils import plotBatch, loss_function, compute_accuracies, CustomLogger
from datetime import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
root = '/home/manuel/ae-flow/src/data/chest_xray'
#root = "./data/chest_xray"
batch_size = 16
epochs = 1000

dataset = AEFlowDataset(root=root, train=True,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
test_dataset = AEFlowDataset(root=root, train=False,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model_name = 'fast_flow'
#model_name = 'res_net'
model = AEFlow(model_name, device)
optimizer = torch.optim.Adam(model.parameters())
alpha = 0.5
beta = 0.9
global_train_step = 0
global_test_step = 0
log_frequency = 10
logger = CustomLogger(root + '/runs/' + model_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

thresholds = np.array([0.0001, 0.001,0.01,0.1, 0.2, 0.3])
PATH = "/home/manuel/ae-flow/src/data/chest_xray/trained-model.pch"

model.load_state_dict(torch.load(PATH))

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for i, (x, y) in enumerate(dataloader):
        if i > 10:
            break
        optimizer.zero_grad()
        x = x.to(device)
        x_prim, log_prob, logdet_jac = model(x)
        #y, log_prob, logdet_jac = torch.utils.checkpoint.checkpoint(model, x)
        loss = loss_function(x, x_prim, log_prob.mean(), logdet_jac.mean(), alpha = 1/2,SSIM = False)
        epoch_loss += loss.item()

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        global_train_step += 1
        anomaly_score = beta * (-torch.exp(log_prob).mean()) + (1 - beta) * -ssim(x_prim, x, reduction='elementwise_mean')
        print(f"Train: epoch {epoch}, iteration: {global_train_step}, anomaly_score : {anomaly_score.item()} train loss = {loss.item()},")
        if global_train_step % log_frequency == 0:
            residual = x - x_prim
            logger.log_all(global_train_step, loss.item() , anomaly_score.item(), x.detach(), x_prim.detach(), residual, mode = 'train')
        torch.cuda.empty_cache() 
    
    accs = np.zeros_like(thresholds)

    if epoch % 2 == 0:
        model.eval()
        test_epoch_loss = 0
        epoch_anomaly_score = 0
        for i, (x, y) in enumerate(test_dataloader):
            if i > 25:
                break
            x = x.to(device)
            x_prim, log_prob, logdet_jac = model(x)

            loss = loss_function(x, y, log_prob.mean(), logdet_jac.mean(), alpha = 1/2,SSIM = False)
            epoch_loss += loss.item()

            anomaly_score = beta * (-torch.exp(log_prob).sum()) + (1 - beta) * -ssim(x_prim, x, reduction='sum')
            epoch_anomaly_score += anomaly_score
            global_test_step += 1
            
            labels, acc = compute_accuracies(thresholds, anomaly_score, y)
            accs += acc
            
            print(f"Test: epoch {epoch}, iteration: {global_test_step} train loss = {loss}, anomaly score= {anomaly_score}, y = {y} accs = {accs/i}")
            
            if global_test_step % log_frequency == 0:
                residual = x - x_prim
                logger.log_all(global_test_step, loss.item() , anomaly_score.item(), x.detach(), x_prim.detach(), residual, mode = 'test')
                logger.log_accuracy(accs/i, thresholds, global_test_step)
            torch.cuda.empty_cache()
            
        print(f"Test: epoch {epoch}, test loss = {test_epoch_loss/len(test_dataset)},"
              f" score  = {epoch_anomaly_score/len(test_dataset)}")
    print(f"Train: epoch {epoch}, test loss = {epoch_loss/len(dataset)}")
    torch.save(model.state_dict(), PATH)
            