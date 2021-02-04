import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, base_distr_dim, embedding_config, generator_lstm_hidden,
                 generator_fc_hidden, leaky_relu_slope, device):
        """"
        Set up the Generator for the model

        Parameters
        ----------
        base_distr_dim: int, the dimension of the base distribution to sample from which has to be transferred
        embedding_config: array, containing number of neurons per layer for embedding the data
        generator_lstm_hidden: int, number of features in the hidden state of the LSTM layer
        generator_fc_hidden: array, number of units per layer to decode the LSTM layer
        leaky_relu_slope: float, slope for the negative part of leaky relu activation
        device: torch.device, cuda if avaiable else cpu

        Returns
        -------
        """
        super(Generator, self).__init__()
        self.device = device

        # set up embedding function
        representation = [nn.Linear(2, embedding_config[0]), nn.LeakyReLU(leaky_relu_slope)]
        for layer_id in range(1, len(embedding_config)):
            representation.extend([nn.Linear(embedding_config[layer_id-1], embedding_config[layer_id]),
                                   nn.LeakyReLU(leaky_relu_slope)])
        self.embedding = nn.Sequential(*representation)

        # set up LSTM layer
        self.generator_lstm_hidden = generator_lstm_hidden
        self.lstm = nn.LSTM(embedding_config[-1], generator_lstm_hidden, batch_first=True)

        # set up the fully connected block serving as decoder
        fc_block = [nn.Linear(generator_lstm_hidden + base_distr_dim, generator_fc_hidden[0]),
                     nn.LeakyReLU(leaky_relu_slope)]
        for layer_id in range(1, len(generator_fc_hidden)):
            fc_block.extend([nn.Linear(generator_fc_hidden[layer_id-1], generator_fc_hidden[layer_id]),
                             nn.LeakyReLU(leaky_relu_slope)])
        fc_block.append(nn.Linear(generator_fc_hidden[-1], 2))
        self.fc = nn.Sequential(*fc_block)

    def forward(self, trajectory, base_distr, traj_lengths):
        """
        Generate the predictions of the next step depending on the sub-trajectories

        Parameters
        ----------
        trajectory: (batch_size, num_subtraj, 2) tensor, the sub-trajectories encountered
            batch_size -> number of samples encountered in one batch
            num_subtraj -> the number of sub-trajectories created from one trajectory
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at a time step
        base_distr: (batch_size, base_distr_dim) tensor, the sample from the base distr which has to be transformed
            batch_size -> number of samples encountered in one batch
            base_distr_dim: int, the dimension of the base distribution to sample from, transforming to next step
        traj_lengths: list of length batch_size containing the length of the sub-trajectories

        Returns
        -------
        next_step: (batch_size, 2) tensor, the next steps for the given sub-trajectories
            batch_size -> number of samples encountered in one batch
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at the next time step
        """
        batch_size = trajectory.shape[0]
        last_indices = [[i for i in range(batch_size)], (np.array(traj_lengths) - 1)]

        h_init, c_init = (torch.zeros((1, batch_size, self.generator_lstm_hidden)).to(self.device) for _ in range(2))
        lstm_out, _ = self.lstm(self.embedding(trajectory), (h_init, c_init))
        lstm_out_last = lstm_out[last_indices]

        lstm_aug = torch.cat((lstm_out_last, base_distr), dim=1)
        next_step = self.fc(lstm_aug) + trajectory[last_indices]
        return next_step
