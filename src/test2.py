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
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
root = '/Vrac/chest_xray'
PATH = root + '/trained-model-fast_flow.pch'
batch_size = 16
epochs = 1000

test_dataset = AEFlowDataset(root=root, train=False,
                        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
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

threshold = -0.3

model.load_state_dict(torch.load(PATH))

model.eval()
test_epoch_loss = 0
epoch_anomaly_score = 0
scores1 = 0.
scores0 = 0.
y1 = 0.
y0 = 0.
acc = 0.
j = 0
anom0 = []
anom1 = []
recon0 = []
recon1 = []
flow0 = []
flow1 = []
for i, (x, y) in tqdm(enumerate(test_dataloader)):
	j+=1
	#if i == 50: break
	with torch.no_grad():
		x = x.to(device)
		y = y.to(device)
		x_prim, log_prob, logdet_jac = model(x)
		flow = beta * (-torch.exp(log_prob.mean()/ (np.log(2) *262144)))
		recon = (1 - beta) * - ssim(x.detach(), x_prim.detach(), reduction='elementwise_mean')
		
		anomaly_score = flow + recon
		
		
		if y == 1:
			
			scores1 += anomaly_score
			y1  +=1
			anom1.append(anomaly_score.item())
			recon1.append(recon.item())
			flow1.append(flow.item())
		else:
			scores0 += anomaly_score
			y0 += 1
			anom0.append(anomaly_score.item())
			recon0.append(recon.item())
			flow0.append(flow.item())
			


print(f'scores1 : {scores1/y1} y1 = {y1}')

print(f'scores0 : {scores0/y0} y0 = {y0}')

plt.hist(anom1, alpha=0.5, label = 'anomalous')
plt.hist(anom0, alpha=0.7, label = 'normal')
"""
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.hist(anom1, alpha=0.5, label = 'anomalous')
ax1.hist(anom0, alpha=0.7, label = 'normal')

ax2.hist(recon1, alpha=0.5, label = 'anomalous')
ax2.hist(recon0, alpha=0.7, label = 'normal')

ax3.hist(flow1, alpha=0.5, label = 'anomalous')
ax3.hist(flow0, alpha=0.7, label = 'normal')
"""
plt.legend()
plt.show()
