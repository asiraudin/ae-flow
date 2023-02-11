
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def loss_function(x,x_prim,log_prob_z_prim,log_det_jac,alpha = 1/2,ssim = False):
    if ssim:
        return (1-alpha)*SSIM(x,x_prim) + alpha*(-log_prob_z_prim-log_det_jac)
    else : 
        return (1-alpha)*nn.functional.mse_loss(x,x_prim) + alpha*(-log_prob_z_prim-log_det_jac)

class State:
    def __init__(self, model, optim) -> None:
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0


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
    
def compute_accuracies(thresholds, anomaly_score, y):
    labels = np.array([int(anomaly_score > t) for t in thresholds])
    accs = np.array([int(l == y) for l in labels])
    return labels, accs
    
class CustomLogger():
    def __init__(self, dir: str) -> None:
        self.logger = SummaryWriter(dir)
    
    def log_all(self, step, loss, anomaly, x, x_recon, residual, mode = 'train'):
        # TODO: Code Residual
        self.logger.add_scalar('loss_' + mode, loss, global_step=step)
        self.logger.add_scalar('anomaly_' + mode, anomaly, global_step=step)
        
        x_image = x[0:3]
        x_recon_image = x_recon[0:3]
        images = torch.cat((x_image, x_recon_image, residual[0:3]), 0)
        grid = make_grid(images, nrow = 3, normalize=True, scale_each = True)
        
        self.logger.add_image('images_' + mode, grid, step)
        
    def log_accuracy(self, accs, thresholds, step):
        print(accs)
        for i, t in enumerate(thresholds):
            self.logger.add_scalar("acc_" + str(t), accs[i], step)
        
