import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TreeLM(nn.Module):
    def __init__(self):
        super(TreeLM, self).__init__()

    def forward(self, seq, hidden):
        """
            seq: [batch#, token#]
            for each batch:
                generate next token
        """
        seq = embedding_lookup(x)

        return x
    
    def embedding_lookup(self, seq):

        return embedding_seq
