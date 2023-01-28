from data.datasets import *
from models.aeflow import *
from utils import plotBatch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


# Load datasets
root = '/home/manuel/ae-flow/src/data/chest_xray'
train_dataset = AEFlowDataset(root, train = True, transform=None)
test_dataset = AEFlowDataset(root, train = False, transform=None)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

channels = 3
im_size = [256, 256]
device = 'cuda'

aeflow = AEFlow(channels, im_size, 'res_net').to(device)



criterion = nn.MSELoss()
optim = torch.optim.Adam(aeflow.parameters())
NB_ITERATIONS = 10

for epoch in range(NB_ITERATIONS): 
    a = 0
    loss_t = 0
    for x,y in train_dataloader:
        x = x.to(device)
        #y = np.random.randint(0,2)
        y = torch.tensor(y).to(device)
    
        optim.zero_grad()
        
        pred = aeflow(x)
        plotBatch(x)
        loss = criterion(pred, y)
        loss_t += loss
       
        loss.backward()
        optim.step()
        
        if a%20 == 0:
            print(f"Itérations {epoch}: loss {loss_t/a}")