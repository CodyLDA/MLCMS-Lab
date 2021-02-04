import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from util.helper import bce_loss
from util.debug_utils import Logger
from time import process_time
from generator import Generator
from discriminator import Discriminator


class TrajectoryGAN:
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.base_distr_dim = config['TrajectoryGAN']['BaseDistrDim']
        self.num_epochs = config['TrajectoryGAN']['NumEpoch']

        # The hyperparameters
        self.improved_steps = config['TrajectoryGAN']['ImprovedDiscriminator']
        generator_fc_hidden = config['TrajectoryGAN']['GeneratorHidden']
        generator_lstm_hidden = config['TrajectoryGAN']['GeneratorLSTM']
        generator_learning_rate = config['TrajectoryGAN']['GeneratorLearning']

        discriminator_fc_hidden = config['TrajectoryGAN']['DiscriminatorHidden']
        discriminator_lstm_hidden = config['TrajectoryGAN']['DiscriminatorLSTM']
        discriminator_learning_rate = config['TrajectoryGAN']['DiscriminatorLearning']

        embedding_config = config['TrajectoryGAN']['Embedding']
        leaky_relu_slope = config['TrajectoryGAN']['LeakyRelu']

        beta_1 = config['TrajectoryGAN']['Beta1']
        beta_2 = config['TrajectoryGAN']['Beta2']

        self.batch_size = config['TrajectoryGAN']['BatchSize']

        self.G = Generator(self.base_distr_dim, embedding_config, generator_lstm_hidden,
                           generator_fc_hidden, leaky_relu_slope, self.device).to(self.device)
        self.D = Discriminator(embedding_config, discriminator_lstm_hidden,
                               discriminator_fc_hidden, leaky_relu_slope, self.device).to(self.device)
        self.G_optimizer = opt.Adam(self.G.parameters(), lr=generator_learning_rate, betas=(beta_1, beta_2))
        self.D_optimizer = opt.Adam(self.D.parameters(), lr=discriminator_learning_rate, betas=(beta_1, beta_2))
        self.mse_loss = nn.MSELoss()

        self.data = {'real_traj': [], 'traj_lengths': [], 'prediction': []}
        self.num_data, self.num_batches, self.start_epoch = 0, 0, 1

    def read_data(self, dataset_path, training_data=True):
        data_files = glob.glob(os.path.dirname(__file__) + dataset_path + "/*.npy")
        training_test_split = int(9 / 10 * len(data_files))
        if not training_data:
            data_files = data_files[training_test_split:]
        else:
            data_files = data_files[:training_test_split]
        dataArr = []
        num, max_traj_length = 0, 0
        for dataFile in data_files:
            file_content = list(np.load(dataFile))
            max_traj_length = len(file_content) if len(file_content) > max_traj_length else max_traj_length

        for dataFile in data_files:
            num += 1
            file_content = list(np.load(dataFile))
            while len(file_content) < max_traj_length:
                file_content.append([file_content[-1][0], file_content[-1][1]])
            dataArr.append(file_content)
            if num % 100 == 0:
                print(f'loaded {num} data points')
        loaded_data = np.asarray(dataArr)

        return loaded_data, max_traj_length

    def load_dataset(self, dataset_path):
        dataset, max_traj_length = self.read_data(dataset_path)
        step_by_step_traj, traj_lengths, next_step_predictions = [], [], []
        for index, traj in enumerate(dataset):
            for step in range(1, len(traj)):
                step_by_step_traj.append(torch.tensor(traj[:step]).float())
                next_step_predictions.append(torch.tensor(traj[step]).float())
                traj_lengths.append(step)

        next_step_predictions = torch.stack(next_step_predictions, dim=0).to(self.device)
        step_by_step_traj = pad_sequence(step_by_step_traj, batch_first=True).to(self.device)

        self.num_data = len(next_step_predictions)
        self.num_batches = int(np.ceil(self.num_data / self.batch_size))
        batch_size = self.batch_size
        for batch_index in range(self.num_batches):
            self.data['real_traj'].append(step_by_step_traj[batch_index * batch_size:
                                                            min((batch_index + 1) * batch_size, self.num_data)])
            self.data['traj_lengths'].append(traj_lengths[batch_index * batch_size:
                                                          min((batch_index + 1) * batch_size, self.num_data)])
            self.data['prediction'].append(next_step_predictions[batch_index * batch_size:
                                                                 min((batch_index + 1) * batch_size, self.num_data)])

    def load_model(self, saving_path):
        self.start_epoch = 1
        assert os.path.isfile(saving_path), f'No model under {saving_path} found!'
        print('Load model from ' + saving_path)
        save_point = torch.load(saving_path, map_location=self.device)
        self.start_epoch = save_point['epoch'] + 1
        self.G.load_state_dict(save_point['G_dict'])
        self.D.load_state_dict(save_point['D_dict'])
        self.G_optimizer.load_state_dict(save_point['G_optimizer'])
        self.D_optimizer.load_state_dict(save_point['D_optimizer'])

    def save_model(self, saving_path, model_name, epoch):
        saving_path_name = f'{saving_path}/{model_name}_{epoch}.pt'
        with open(saving_path_name, 'w') as f:
            pass
        logger.print_me('Saving model to ', saving_path_name)
        torch.save({
            'epoch': epoch,
            'G_dict': self.G.state_dict(),
            'D_dict': self.D.state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict()
        }, saving_path_name)

    def train_batch(self, real_traj, prediction, traj_lengths):
        batch_size = len(prediction)

        zeros = torch.zeros(batch_size, 1).to(self.device)
        ones = torch.ones(batch_size, 1).to(self.device)
        base_distr = torch.rand(batch_size, self.base_distr_dim).float().to(self.device)
        zeros.requires_grad, ones.requires_grad, base_distr.requires_grad = False, False, False

        self.D_optimizer.zero_grad()
        for improvement in range(self.improved_steps + 1):
            with torch.no_grad():
                prediction_fake = self.G(real_traj, base_distr, traj_lengths)

            fake_labels = self.D(real_traj, prediction_fake, traj_lengths)
            d_loss = bce_loss(fake_labels, zeros)

            real_labels = self.D(real_traj, prediction, traj_lengths)
            d_loss += bce_loss(real_labels, ones)

            d_loss.backward()
            self.D_optimizer.step()

        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        prediction_fake = self.G(real_traj, base_distr, traj_lengths)

        fake_labels = self.D(real_traj, prediction_fake, traj_lengths)
        g_loss = bce_loss(fake_labels, ones)
        mse_loss = 100 * F.mse_loss(prediction_fake, prediction)
        g_loss += mse_loss

        g_loss.backward()
        self.G_optimizer.step()

        return g_loss.item(), d_loss.item(), mse_loss.item()

    def train(self, saving_path, model_name):
        for epoch in range(self.start_epoch, self.num_epochs):
            g_loss, d_loss, mse_loss = 0, 0, 0

            tic = process_time()
            for index in range(self.num_batches):
                g_loss_index, d_loss_index, mse_loss_index = self.train_batch(self.data['real_traj'][index],
                                                                              self.data['prediction'][index],
                                                                              self.data['traj_lengths'][index])
                g_loss += g_loss_index
                d_loss += d_loss_index
                mse_loss += mse_loss_index
            toc = process_time()

            logger.print_me(f'#{epoch:5d} | MSE = {mse_loss:.5f} | Loss G = {g_loss:.4f} '
                            f'| Loss D = {d_loss:.4f} | time = {toc - tic:.2f} s')

            if epoch % 50 == 0:
                self.save_model(saving_path, model_name, epoch)

    def generate(self, starting_points, num_step):
        past_traj = starting_points.data.unsqueeze(1).to(self.device)
        num_samples = starting_points.shape[0]

        for step in range(1, num_step + 1):
            base_distr = torch.rand(num_samples, self.base_distr_dim).float().to(self.device)
            base_distr.requires_grad = False

            next_step = self.G(past_traj, base_distr, [step] * num_samples)
            past_traj = torch.cat([past_traj, next_step.data.unsqueeze(1)], dim=1)

        print(f'Generated {num_samples} new trajectories of length {num_step}.')
        return past_traj.cpu().numpy()


if __name__ == '__main__':
    # Read config file
    config_file = 'config/config.yaml'
    f = open(config_file)
    conf = yaml.load(f, Loader=yaml.FullLoader)

    saving_path = os.path.dirname(__file__) + '/' + conf['TrajectoryGAN']['SavingPath']
    model_name = conf['TrajectoryGAN']['ModelName']

    gan = TrajectoryGAN(conf)

    training = conf['TrajectoryGAN']['Train']

    if training:
        logger = Logger(f'{conf["Logger"]}_{model_name}.txt')
        # Train
        gan.load_dataset(conf['DatasetPath'])
        gan.train(saving_path, model_name)
    else:
        # Generate
        gan.load_model(f"{saving_path}/{conf['TrajectoryGAN']['ModelName']}.pt")
        data, steps = gan.read_data(conf['DatasetPath'], False)
        data = data[:20, :, :]
        n_samples = len(data)
        data_start = data[-n_samples:, 0, :]
        data = data[-n_samples:, :, :]
        trajectories = gan.generate(torch.from_numpy(data_start).float(), steps)

        fig, ax = plt.subplots(2)
        for i in range(n_samples):
            ax[0].plot(trajectories[i, :, 0], trajectories[i, :, 1])
            ax[1].plot(data[i, :, 0], data[i, :, 1])

        ax[0].set_title('Generated')
        ax[1].set_title('Real')
        fig.show()
