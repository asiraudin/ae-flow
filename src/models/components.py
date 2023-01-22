import torch
import torch.nn as nn


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
