import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2 as wide_resnet
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = wide_resnet(pretrained=True)
        resnet_modules = list(resnet.children())[:-3]
        self.encoder = nn.Sequential(*resnet_modules)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


def get_upscaling_block(channels_in, channels_out, last_layer=False):
    '''
    Each transpose conv will be followed by BatchNorm and ReLU,
    except the last block (which is only followed by tanh)
    '''
    if last_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, (2, 2), 2, bias=False),
            nn.Tanh()
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_in, (2, 2), 2, bias=False),
            torch.nn.BatchNorm2d(num_features=channels_in),
            nn.ReLU(),
            nn.Conv2d(channels_in, channels_out, (3, 3), 1, 1),
            torch.nn.BatchNorm2d(num_features=channels_out),
            nn.ReLU()
        )


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = get_upscaling_block(1024, 512)
        self.block2 = get_upscaling_block(512, 256)
        self.block3 = get_upscaling_block(256, 64)
        self.block4 = get_upscaling_block(64, 3, last_layer=True)

        self.decoder = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4
        )

    def forward(self, x):
        return self.decoder(x)


class Flow(nn.Module):
    def __init__(self, channels, dims_in, n_blocks):
        super().__init__()

        self.inn = SequenceINN(channels, *dims_in)
        for i in range(n_blocks):
            self.inn.append(AllInOneBlock, subnet_constructor=self.build_flow_subnet)

    def forward(self, x, rev=False):
        """
        Output latent variable + log jacobian determinant. If rev = True, use flow in backward mode.

        :param x: input data
        :param rev: bool - whether or not to use the flow in backward mode
        :return: torch.tensor, float
        """
        out, jac = self.inn(x, rev=rev)
        return out, jac

    @staticmethod
    def build_flow_subnet():
        """
        Static method that build the subnet used in flow block.
        :return: nn.Sequential
        """
        return None
