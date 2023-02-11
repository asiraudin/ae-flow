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
device = "cpu"
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

model.eval()
test_epoch_loss = 0
epoch_anomaly_score = 0
scores1 = 0.
scores0 = 0.
y1 = 0.
y0 = 0.
for i, (x, y) in enumerate(test_dataloader):
    if i% 100 == 99:
        break
    x = x.to(device)
    y = y.to(device)
    x_prim, log_prob, logdet_jac = model(x)

    loss = loss_function(x, y, log_prob.mean(), logdet_jac.mean(), alpha = 1/2,SSIM = False)


    anomaly_score = beta * (-torch.exp(log_prob.detach()).mean()) + (1 - beta) * - ssim(x.detach(), x_prim.detach(), reduction='sum')
    epoch_anomaly_score += anomaly_score
    if y == 1:
        scores1 += anomaly_score
        y1  +=1
    else:
        scores0 += anomaly_score
        y0 += 1
    
    #labels, acc = compute_accuracies(thresholds, anomaly_score, y)
    #accs += acc
    
    torch.cuda.empty_cache()

print(f'scores1 : {scores1/y1}')

print(f'scores0 : {scores0/y0}')