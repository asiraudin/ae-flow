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

#model_name = 'fast_flow'
model_name = 'res_net'
#PATH = root + '/trained-model-'+model_name+'.pch'
model = AEFlow(model_name, device)
optimizer = torch.optim.Adam(model.parameters())
alpha = 0.5
beta = 0.9
global_train_step = 0
global_test_step = 0
log_frequency = 10
logger = CustomLogger(root + '/runs/' + model_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

threshold = -0.3
root = '/Vrac/chest_xray'
PATH = root + '/trained-model-res_net.pch'

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
for i, (x, y) in tqdm(enumerate(test_dataloader)):
	j+=1
	#if i == 50: break
	with torch.no_grad():
		x = x.to(device)
		y = y.to(device)
		x_prim, log_prob, logdet_jac = model(x)

		
		anomaly_score = beta * (-torch.exp(log_prob.mean()/ (np.log(2) *262144))) + (1 - beta) * - ssim(x.detach(), x_prim.detach(), reduction='elementwise_mean')

		
		if y == 1:
			
			scores1 += anomaly_score
			y1  +=1
			anom1.append(anomaly_score.item())
		else:
			scores0 += anomaly_score
			y0 += 1
			anom0.append(anomaly_score.item())
			


print(f'scores1 : {scores1/y1} y1 = {y1}')

print(f'scores0 : {scores0/y0} y0 = {y0}')


plt.hist(anom1, alpha=0.5)
plt.hist(anom0, alpha=0.7)
plt.show()
