import numpy as np
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, embedding_config, discriminator_lstm_hidden, discriminator_fc_hidden, leaky_relu_slope, device):
        super(Discriminator, self).__init__()
        self.device = device

        representation = [nn.Linear(2, embedding_config[0]), nn.LeakyReLU(leaky_relu_slope),
                          nn.BatchNorm1d(embedding_config[0])]
        for layer_id in range(1, len(embedding_config)):
            representation.extend([nn.Linear(embedding_config[layer_id - 1], embedding_config[layer_id]),
                                   nn.LeakyReLU(leaky_relu_slope)])
        self.embedding = nn.Sequential(*representation)

        self.discriminator_lstm_hidden = discriminator_lstm_hidden
        self.lstm = nn.LSTM(embedding_config[-1], discriminator_lstm_hidden, batch_first=True)

        fc_block = [nn.Linear(discriminator_lstm_hidden + embedding_config[-1], discriminator_fc_hidden[0]),
                    nn.LeakyReLU(leaky_relu_slope)]
        for layer_id in range(1, len(discriminator_fc_hidden)):
            fc_block.extend([nn.Linear(discriminator_fc_hidden[layer_id - 1], discriminator_fc_hidden[layer_id]),
                             nn.LeakyReLU(leaky_relu_slope)])
        fc_block.append(nn.Linear(discriminator_fc_hidden[-1], 1))
        self.fc = nn.Sequential(*fc_block)

    def forward(self, trajectory, real_fake_pred, traj_lengths):
        batch_size = trajectory.size(0)
        last_indices = [[i for i in range(batch_size)], (np.array(traj_lengths) - 1)]

        embedded_traj = self.embedding(trajectory.view(-1, 2)).view(batch_size, trajectory.size(1), -1)
        embedded_pred = self.embedding(real_fake_pred)

        h_init, c_init = (torch.zeros((1, batch_size, self.discriminator_lstm_hidden)).to(self.device)
                          for _ in range(2))
        lstm_out, _ = self.lstm(embedded_traj, (h_init, c_init))
        lstm_out_last = lstm_out[last_indices]

        lstm_aug = torch.cat((lstm_out_last, embedded_pred), dim=1)
        decision = self.fc(lstm_aug)
        return decision
