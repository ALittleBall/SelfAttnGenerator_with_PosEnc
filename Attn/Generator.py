import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.autograd import  Variable
from AttLayer import Self_Attn

class Generator(nn.Module):
    def __init__(self,use_spectral_norm=True):
        super(Generator, self).__init__()

        self.enc_layer1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=7),
            nn.ReLU(True))

        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.enc_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))
            
        self.att1 = Self_Attn(64)
        
        self.att2 = Self_Attn(64)

        self.dec_layer4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.dec_layer2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(32, track_running_stats=False),
            nn.ReLU(True))
        
        self.dec_layer1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7),
        )

    def forward(self, x):
        enc_out1 = self.enc_layer1(x)
        enc_out2 = self.enc_layer2(enc_out1)
        enc_out4 = self.enc_layer4(enc_out2)

        attout = self.att1(enc_out4)
        attout = self.att2(attout)

        dec_out4 = self.dec_layer4(attout)
        
        dec_out2 = self.dec_layer2(dec_out4)
        
        dec_out1 = self.dec_layer1(dec_out2)
        x = torch.sigmoid(dec_out1)
        return x