import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torchvision.models import wide_resnet50_2 as wide_resnet, Wide_ResNet50_2_Weights
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = wide_resnet(weights=Wide_ResNet50_2_Weights.DEFAULT)
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


def build_fast_flow(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_in, (3, 3), padding="same"),
        nn.ReLU(),
        nn.Conv2d(channels_in, channels_out, (1, 1), padding="same"),
        )


def build_res_net(channels_in, channels_out):
    return ResNet(channels_in, channels_out)



class ResNet(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, (3, 3), padding="same", bias=False),
            nn.BatchNorm2d(num_features=channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, (3, 3), padding="same", bias=False),
            nn.BatchNorm2d(num_features=channels_out)
            )
        
        if channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, (1, 1), bias=False),
                nn.BatchNorm2d(channels_out)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.layers(x)
        out = out + self.shortcut(x)
        #out = nn.functional.relu(out)
        return out
        
        
        
class Flow(nn.Module):
    def __init__(self, channels, dims_in, n_blocks, subnet_type="fast_flow", device = 'cpu'):
        super().__init__()

        if subnet_type == "fast_flow":
            subnet_constructor = build_fast_flow
        elif subnet_type == "res_net":
            subnet_constructor = build_res_net
        else:
            raise NotImplementedError

        self.inn = SequenceINN(channels, *dims_in)
        for i in range(n_blocks):
            self.inn.append(AllInOneBlock, subnet_constructor=subnet_constructor)
        self.prior = Independent(Normal(torch.zeros(channels*dims_in[0]*dims_in[1]).to(device),
                                        torch.ones(channels*dims_in[0]*dims_in[1]).to(device)), 1)

    def forward(self, x, rev=False):
        """
        Output latent variable + log jacobian determinant. If rev = True, use flow in backward mode.

        :param x: input data
        :param rev: bool - whether or not to use the flow in backward mode
        :return: torch.tensor, float
        """
        z, jac = self.inn(x, rev=rev)
        logprob_z = self.prior.log_prob(z.flatten(start_dim=1))
        return z, logprob_z, jac
