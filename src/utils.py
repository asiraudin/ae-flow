
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as SSIM

def loss_function(x,x_prim,log_prob_z_prim,log_det_jac,alpha = 1/2,SSIM = False):
    if SSIM:
        return (1-alpha)*SSIM(x,x_prim) + alpha*(-log_prob_z_prim-log_det_jac)
    else : 
        return (1-alpha)*nn.functional.mse_loss(x,x_prim) + alpha*(-log_prob_z_prim-log_det_jac)




def plotBatch(batch):
    batch_size = batch.shape[0]
    columns = int(np.ceil(np.sqrt(batch_size)))
    rows = columns
    fig = plt.figure()
    for i in range(batch_size):
        img = batch[i].cpu().detach().numpy().T
        img = (img - np.min(img))/(np.max(img)-np.min(img))
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()