import numpy as np
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, embedding_config, discriminator_lstm_hidden, discriminator_fc_hidden, leaky_relu_slope, device):
        """"
        Set up the Discriminator for the model

        Parameters
        ----------
        embedding_config: array, containing number of neurons per layer for embedding the data
        discriminator_lstm_hidden: int, number of features in the hidden state of the LSTM layer
        discriminator_fc_hidden: array, number of units per layer to decode the LSTM layer
        leaky_relu_slope: float, slope for the negative part of leaky relu activation
        device: torch.device, cuda if avaiable else cpu

        Returns
        -------
        """
        super(Discriminator, self).__init__()
        self.device = device

        # setting up the embedding function
        representation = [nn.Linear(2, embedding_config[0]), nn.LeakyReLU(leaky_relu_slope),
                          nn.BatchNorm1d(embedding_config[0])]
        for layer_id in range(1, len(embedding_config)):
            representation.extend([nn.Linear(embedding_config[layer_id - 1], embedding_config[layer_id]),
                                   nn.LeakyReLU(leaky_relu_slope)])
        self.embedding = nn.Sequential(*representation)

        # set up the LSTM layer
        self.discriminator_lstm_hidden = discriminator_lstm_hidden
        self.lstm = nn.LSTM(embedding_config[-1], discriminator_lstm_hidden, batch_first=True)

        # setting up the fully connected block for decoding
        fc_block = [nn.Linear(discriminator_lstm_hidden + embedding_config[-1], discriminator_fc_hidden[0]),
                    nn.LeakyReLU(leaky_relu_slope)]
        for layer_id in range(1, len(discriminator_fc_hidden)):
            fc_block.extend([nn.Linear(discriminator_fc_hidden[layer_id - 1], discriminator_fc_hidden[layer_id]),
                             nn.LeakyReLU(leaky_relu_slope)])
        fc_block.append(nn.Linear(discriminator_fc_hidden[-1], 1))
        self.fc = nn.Sequential(*fc_block)

    def forward(self, trajectory, real_fake_pred, traj_lengths):
        """
        Discriminate the predictions and classify the predictions as being real or fake

        Parameters
        ----------
        trajectory: (batch_size, num_subtraj, 2) tensor, the sub-trajectories encountered
            batch_size -> number of samples encountered in one batch
            num_subtraj -> the number of sub-trajectories created from one trajectory
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at a time step
        real_fake_pred: (batch_size, 2) tensor, either real or fake predictions for the next step
            batch_size -> number of samples encountered in one batch
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at the next time step
        traj_lengths: list of length batch_size containing the length of the sub-trajectories

        Returns
        -------
        decision: (batch_size, 1) tensor, the decision if real or fake
            batch_size -> number of samples encountered in one batch
            1 -> last dimension contains a scalar determining classification of real or fake
        """
        batch_size = trajectory.shape[0]
        last_indices = [[i for i in range(batch_size)], (np.array(traj_lengths) - 1)]

        # embedd the sub-trajectories
        embedded_traj = self.embedding(trajectory.view(-1, 2)).view(batch_size, trajectory.size(1), -1)
        embedded_pred = self.embedding(real_fake_pred)

        # feed through LSTM-layer
        h_init, c_init = (torch.zeros((1, batch_size, self.discriminator_lstm_hidden)).to(self.device)
                          for _ in range(2))
        lstm_out, _ = self.lstm(embedded_traj, (h_init, c_init))
        lstm_out_last = lstm_out[last_indices]

        # decode the information
        lstm_aug = torch.cat((lstm_out_last, embedded_pred), dim=1)
        decision = self.fc(lstm_aug)
        return decision
