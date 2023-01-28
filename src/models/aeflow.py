import torch.nn as nn
from .components import Decoder, Encoder, Flow


class AEFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.flow = Flow(1024, (16, 16), 8)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x, log_prob, jac = self.flow(x)
        x = self.decoder(x)
        return x, log_prob, jac


