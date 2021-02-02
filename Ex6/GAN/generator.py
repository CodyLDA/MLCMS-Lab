import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, base_distr_dim, embedding_config, generator_lstm_hidden,
                 generator_fc_hidden, leaky_relu_slope, device):
        super(Generator, self).__init__()
        self.device = device

        representation = [nn.Linear(2, embedding_config[0]), nn.LeakyReLU(leaky_relu_slope)]
        for layer_id in range(1, len(embedding_config)):
            representation.extend([nn.Linear(embedding_config[layer_id-1], embedding_config[layer_id]),
                                   nn.LeakyReLU(leaky_relu_slope)])
        self.embedding = nn.Sequential(*representation)

        self.generator_lstm_hidden = generator_lstm_hidden
        self.lstm = nn.LSTM(embedding_config[-1], generator_lstm_hidden, batch_first=True)

        fc_block = [nn.Linear(generator_lstm_hidden + base_distr_dim, generator_fc_hidden[0]),
                     nn.LeakyReLU(leaky_relu_slope)]
        for layer_id in range(1, len(generator_fc_hidden)):
            fc_block.extend([nn.Linear(generator_fc_hidden[layer_id-1], generator_fc_hidden[layer_id]),
                             nn.LeakyReLU(leaky_relu_slope)])
        fc_block.append(nn.Linear(generator_fc_hidden[-1], 2))
        self.fc = nn.Sequential(*fc_block)

    def forward(self, trajectory, base_distr, traj_lengths):
        batch_size = trajectory.shape[0]
        last_indices = [[i for i in range(batch_size)], (np.array(traj_lengths) - 1)]

        h_init, c_init = (torch.zeros((1, batch_size, self.generator_lstm_hidden)).to(self.device) for _ in range(2))
        lstm_out, _ = self.lstm(self.embedding(trajectory), (h_init, c_init))
        lstm_out_last = lstm_out[last_indices]

        lstm_aug = torch.cat((lstm_out_last, base_distr), dim=1)
        next_step = self.fc(lstm_aug) + trajectory[last_indices]
        return next_step
