
from matplotlib import pyplot as plt
import numpy as np



def plotBatch(batch):
    batch_size = batch.shape[0]
    columns = int(np.ceil(np.sqrt(batch_size)))
    rows = columns
    fig = plt.figure()
    for i in range(batch_size):
        img = batch[i].cpu().detach().numpy().T
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()