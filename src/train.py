import torch
import numpy as np

from data.datasets import AEFlowDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import AEFlow
from torchmetrics.functional import structural_similarity_index_measure as ssim
from utils import CustomLogger
from datetime import datetime
from torch import nn
from tqdm import tqdm
import argparse
import os

def train(submodel_name, epochs, model_path, dataset_path):
    # Your training code goes here
    print("Training submodel:", submodel_name)
    print("Number of epochs:", epochs)
    print("Model path:", model_path)
    print("Dataset path:", dataset_path)
    
    current_directory = os.getcwd()
    print("The current directory is:", current_directory)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    dataset = AEFlowDataset(root=dataset_path, train=True,
                            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = AEFlow(submodel_name, device)
    optimizer = torch.optim.Adam(model.parameters())
    alpha = 0.5
    beta = 0.9  
    global_train_step = 0
    log_frequency = 100

    logger = CustomLogger(dataset_path + '/runs/' + submodel_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S"))


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_anomaly = 0.0
        j = 0
        for i, (x, y) in tqdm(enumerate(dataloader)):
            j+=1
            optimizer.zero_grad()
            x = x.to(device)
            x_prim, log_prob, logdet_jac = model(x)
            l_recon = (1-alpha)*nn.functional.mse_loss(x,x_prim)
            l_flow = alpha*(-log_prob.mean() / (np.log(2) *262144) - logdet_jac.mean() / (np.log(2) *262144))
            loss = l_recon + l_flow
            epoch_loss += loss.item()

            loss = loss.mean()
            loss.backward()
            optimizer.step()
            global_train_step += 1
            anomaly_score = beta * (-torch.exp(log_prob).mean()) + (1 - beta) * - ssim(x.detach(), x_prim.detach(), reduction='sum')
            epoch_anomaly += anomaly_score.item()
            
        
            if global_train_step % log_frequency == 0:
                
                residual = x - x_prim
                logger.log_all(global_train_step, loss.item() , anomaly_score.item(), x.detach(), x_prim.detach(), residual, mode = 'train')

            torch.cuda.empty_cache() 
        print(f"Train: epoch {epoch}, anomaly_score : {epoch_anomaly/j} train loss = {epoch_loss/j},")


if __name__ == "__main__":
    current_directory = os.getcwd()
    parser = argparse.ArgumentParser(description='Train a submodel')
    parser.add_argument('--submodel_name', type=str, required=False, default='fast_flow',  help='fast_flow ou res_net, default fast_flow')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='int, default 100')
    parser.add_argument('--model_path', type=str, required=False, default=current_directory + '/trained-model.pch', help='Path to the model file')
    parser.add_argument('--dataset_path', type=str, required=False, default=current_directory + '/data/chest_xray', help='filepath to save the trained model (saved after each epoch), default workdir/trained-model.pch')

    args = parser.parse_args()
    submodel_name = args.submodel_name
    epochs = args.epochs
    model_path = args.model_path
    dataset_path = args.dataset_path

    train(submodel_name, epochs, model_path, dataset_path)
    