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
    def __init__(self, channels, dims_in, subnet_type):
        super().__init__()
        self.encoder = Encoder()
        self.flow = Flow(channels= 1024, dims_in= [16, 16], n_blocks= 1, subnet_type=subnet_type)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.flow(x)
        x = self.decoder(x)
        return x

