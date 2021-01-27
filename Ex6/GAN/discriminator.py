import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import glob
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
from util.parse_utils import BIWIParser
from util.helper import bce_loss
from util.debug_utils import Logger


class Discriminator(nn.Module):
    def __init__(self,  embedding_size, lstm_size, hidden_size, relu_slope):
        super(Discriminator, self).__init__()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Embedding
        embed_layers = [nn.Linear(2, embedding_size[0]), nn.LeakyReLU(relu_slope), nn.BatchNorm1d(embedding_size[0])]
        for ii in range(1, len(embedding_size)):
            embed_layers.extend([nn.Linear(embedding_size[ii - 1], embedding_size[ii]), nn.LeakyReLU(relu_slope)])
        self.embedding = nn.Sequential(*embed_layers)

        # LSTM
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_size[-1], lstm_size, num_layers=1, batch_first=True)

        # Classifier
        fc_layers = [nn.Linear(lstm_size + embedding_size[-1], hidden_size[0]), nn.LeakyReLU(relu_slope)]
        for ii in range(1, len(hidden_size)):
            fc_layers.extend([nn.Linear(hidden_size[ii - 1], hidden_size[ii]), nn.LeakyReLU(relu_slope)])
        fc_layers.append(nn.Linear(hidden_size[-1], 1))
        self.fc = nn.Sequential(*fc_layers)

    def load_backup(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, x_in, x_out, x_lengths):
        bs = x_in.size(0)
        T = x_in.size(1)
        e_in = self.embedding(x_in.view(-1, 2)).view(bs, T, -1)
        e_out = self.embedding(x_out) #TODO but here, x_out is not changed, hence it should be (128, 2)?

        h_init, c_init = (torch.zeros(1, bs, self.lstm_size, device=x_in.device) for _ in range(2))
        lstm_out, (h_out, c_out) = self.lstm(e_in, (h_init, c_init))
        inds = [[i for i in range(bs)], (np.array(x_lengths) - 1)]
        lstm_out_last = lstm_out[inds]

        hid_vector = torch.cat((lstm_out_last, e_out), dim=1)
        x_out = self.fc(hid_vector)
        return x_out
