import torch
import torch.nn as nn
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.autograd import  Variable
import math

class Self_Attn(nn.Module):
    def __init__(self,in_dim, heads = 2):
        super(Self_Attn,self).__init__()


        self.multihead = nn.ModuleList([Attn_sublayer(in_dim) for _ in range (2)])

        self.pos = (pos_embed(64).view(1,4096,4096)).cuda().float()

        self.conv = nn.Conv2d(in_channels = 2*in_dim , out_channels = in_dim , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self,x):
        heads_output = [layers(x, self.pos) for layers in self.multihead]

        head_concate = torch.cat(heads_output, 1)

        out = self.conv(head_concate)

        out = self.gamma * out + x

        return out

# Attn_sublayer

class Attn_sublayer(nn.Module):
    def __init__(self,in_dim):
        super(Attn_sublayer,self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x, pos):
        
        """ 
        The basic calculation formular comes from SAGAN
        https://arxiv.org/abs/1805.08318 
        We creatively encode the position below
        
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        """ SAGAN code ends here """

        # Position Encode:
        attention = attention*pos

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        return out


"""
My way to encode the position in the attention layer
"""
def pos_embed(size):
    pos = torch.zeros((size*size,size*size))
    for i in range(size*size):
        x_center = i % size
        y_center = i // size
        for j in range(size*size):
            x = j % size
            y = j // size
            relative_x = abs(x_center - x)
            relative_y = abs(y_center - y)
            pos[i,j] = gaussian(relative_x, relative_y)
    return pos



def gaussian(x, y):
    sig = 30
    return 1/2/math.pi/(sig**2)*math.e**(-1*1/(2*sig**2)*(x**2+y**2))*2000

