import torch.nn as nn
from .components import Decoder, Encoder, Flow


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class AEFlow(nn.Module):
    def __init__(self, subnet_type, device = 'cpu'):
        super().__init__()
        self.encoder = Encoder().to(device)
        self.flow = Flow(channels= 1024, dims_in= (16, 16), n_blocks= 8, subnet_type=subnet_type, device = device).to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        x = self.encoder(x)
        out, log_prob, jac = self.flow(x)
        out = self.decoder(out)
        return out, log_prob, jac

