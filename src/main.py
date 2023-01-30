from data.datasets import *
from models.aeflow import *
from utils import plotBatch, loss_function
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Load datasets
root = '/home/manuel/ae-flow/src/data/chest_xray'
train_dataset = AEFlowDataset(root, train = True, transform=None)
test_dataset = AEFlowDataset(root, train = False, transform=None)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = 'cpu'

#aeflow = AEFlow('fast_flow', device )
aeflow = AEFlow('res_net', device )


criterion = nn.MSELoss()
optim = torch.optim.Adam(aeflow.parameters())
NB_ITERATIONS = 10

for epoch in range(NB_ITERATIONS): 
    a = 0
    alpha = 0.5
    loss_t = 0
    for x,y in train_dataloader:
        optim.zero_grad()
        x = x.to(device).requires_grad_()
        y = y.to(device)
    
        
        #x_recon, log_prob, jac = torch.utils.checkpoint.checkpoint(aeflow, x)
        x_recon, log_prob, jac = aeflow(x)
        loss = loss_function(x, x_recon, log_prob.mean(), jac.mean(), alpha = 1/2,SSIM = False)
       
        loss.backward()
        optim.step()
        a += 1
        
        print(f"Itérations {epoch}, a = {a}: loss {loss}")
        if a % 10 == 0:
            plotBatch(x_recon)