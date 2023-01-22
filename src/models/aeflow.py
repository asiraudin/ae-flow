import torch.nn as nn
from torchvision.models import wide_resnet50_2 as wide_resnet
from .components import Decoder


class AEFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.setup_encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def setup_encoder(self):
        resnet = wide_resnet(pretrained=True)
        resnet_modules = list(resnet.children())[:-3]
        self.encoder = nn.Sequential(*resnet_modules)
        for p in self.encoder.parameters():
            p.requires_grad = False
