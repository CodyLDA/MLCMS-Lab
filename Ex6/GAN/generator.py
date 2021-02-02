import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_size, lstm_size, hidden_size, relu_slope):
        super(Generator, self).__init__()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Embedding
        embed_layers = [nn.Linear(2, embedding_size[0]), nn.LeakyReLU(relu_slope)]
        for ii in range(1, len(embedding_size)):
            embed_layers.extend([nn.Linear(embedding_size[ii-1], embedding_size[ii]), nn.LeakyReLU(relu_slope)])
        self.embedding = nn.Sequential(*embed_layers)

        # LSTM
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_size[-1], lstm_size, num_layers=1, batch_first=True)

        # Decoder
        fc_layers = [nn.Linear(lstm_size + noise_dim, hidden_size[0]), nn.LeakyReLU(relu_slope)]
        for ii in range(1, len(hidden_size)):
            fc_layers.extend([nn.Linear(hidden_size[ii-1], hidden_size[ii]), nn.LeakyReLU(relu_slope)])
        fc_layers.append(nn.Linear(hidden_size[-1], 2))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x_in, noise, x_lengths):
        bs = noise.size(0)
        last_indices = [[i for i in range(bs)], (np.array(x_lengths) - 1)]

        # calc velocities and concat to x_in
        #x_in_vel = x_in[:, 1:] - x_in[:, :-1]
        #x_in_vel = torch.from_numpy(x_in_vel)
        #x_in_vel = torch.cat((x_in_vel, torch.zeros((bs, 1, 2), device=torch.from_numpy(x_in).device, dtype=torch.from_numpy(x_in).dtype)), dim=1)
        #last_indices_1 = [[i for i in range(bs)], (np.array(x_lengths) - 2)]
        #x_in_vel[last_indices] = x_in_vel[last_indices_1]
        #x_in_aug = torch.cat([torch.from_numpy(x_in), x_in_vel], dim=2)

        #e_in = self.embedding(x_in_aug)
        e_in = self.embedding(x_in)

        h_init, c_init = (torch.zeros((1, bs, self.lstm_size), device=noise.device) for _ in range(2))
        lstm_out, (h_out, c_out) = self.lstm(e_in, (h_init, c_init))
        lstm_out_last = lstm_out[last_indices]

        hid_vector = torch.cat((lstm_out_last, noise), dim=1)
        x_out = self.fc(hid_vector) + x_in[last_indices]
        return x_out
