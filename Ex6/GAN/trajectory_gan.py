import os
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
from torch.nn.utils.rnn import pad_sequence
from util.helper import bce_loss
from util.debug_utils import Logger
from time import process_time
from generator import Generator
from discriminator import Discriminator


class TrajectoryGAN:

    def __init__(self, config):
        self.noise_dim = config['TrajectoryGAN']['BaseDistrDim']
        self.num_epochs = config['TrajectoryGAN']['NumEpoch']
        self.saving_point = config['TrajectoryGAN']['SavingPoint']

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

        self.G = Generator(self.noise_dim, embedding_config, generator_lstm_hidden, generator_fc_hidden, leaky_relu_slope).cuda()
        self.D = Discriminator(embedding_config, discriminator_lstm_hidden, discriminator_fc_hidden, leaky_relu_slope).cuda()
        self.G_optimizer = opt.Adam(self.G.parameters(), lr=generator_learning_rate, betas=(beta_1, beta_2))
        self.D_optimizer = opt.Adam(self.D.parameters(), lr=discriminator_learning_rate, betas=(beta_1, beta_2))
        self.mse_loss = nn.MSELoss()

        self.data = {'obsvs': [], 'obsv_lengths': [], 'preds': []}
        self.n_data = 0
        self.n_batches = 0
        self.test_data_init = []
        self.start_epoch = 1

    def readData(self, datasetPath, training=True):
        dataFiles = glob.glob(os.path.dirname(__file__) + datasetPath + "/*.npy")[:128*8]
        training_test_split = int(9 / 10 * len(dataFiles))
        if not training:
            dataFiles = dataFiles[training_test_split:]
        else:
            dataFiles = dataFiles[:training_test_split]
        dataArr, obsv_lengths = [], []
        num, max = 0, 0
        for dataFile in dataFiles:
            arr = np.load(dataFile).tolist()
            if len(arr) > max:
                max = len(arr)

        for dataFile in dataFiles:
            num += 1
            arr = np.load(dataFile).tolist()
            while len(arr) < max:
                arr.append([arr[-1][0], arr[-1][1]])
            dataArr.append(arr)
            obsv_lengths.append(max)
            if num % 100 == 0:
                print(f'loaded {num} data points')
        arr = np.asarray(dataArr)
        print(arr.shape)
        max_observation_length = max

        return arr, max_observation_length

    def load_dataset(self, datasetPath):
        dataset, max_obs_len = self.readData(datasetPath)
        continuation_data_obsv = []
        obsv_lengths = []
        continuation_data_pred = []
        for ii, Pi in enumerate(dataset):
            for tt in range(1, len(Pi)):
                x_obsv_t = Pi[max(0, tt - max_obs_len):tt]
                obsv_lengths.append(len(x_obsv_t))
                continuation_data_obsv.append(torch.FloatTensor(x_obsv_t))
                continuation_data_pred.append(torch.FloatTensor(Pi[tt]))

        continuation_data_pred = torch.stack(continuation_data_pred, dim=0).cuda()
        continuation_data_obsv = pad_sequence(continuation_data_obsv, batch_first=True).cuda()

        self.n_data = len(continuation_data_pred)
        self.n_batches = int(np.ceil(self.n_data / self.batch_size))
        bs = self.batch_size
        for bi in range(self.n_batches):
            self.data['obsvs'].append(continuation_data_obsv[bi * bs:min((bi + 1) * bs, self.n_data)])
            self.data['obsv_lengths'].append(obsv_lengths[bi * bs:min((bi + 1) * bs, self.n_data)])
            self.data['preds'].append(continuation_data_pred[bi * bs:min((bi + 1) * bs, self.n_data)])

    def save(self, saving_point, epoch):
        logger.print_me('Saving model to ', saving_point)
        torch.save({
            'epoch': epoch,
            'G_dict': self.G.state_dict(),
            'D_dict': self.D.state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict()
        }, saving_point)

    def load_model(self, saving_point='', overwrite=False):
        self.saving_point = os.path.dirname(__file__) + '/' + saving_point
        self.start_epoch = 1
        if os.path.isfile(saving_point):
            if overwrite:
                f = open(saving_point, "w")
                f.close()
                return None
            print('loading from ' + saving_point)
            checkpoint = torch.load(saving_point)
            self.start_epoch = checkpoint['epoch'] + 1
            self.G.load_state_dict(checkpoint['G_dict'])
            self.D.load_state_dict(checkpoint['D_dict'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        else:
            f = open(saving_point, "x")
            f.close()

    def batch_train(self, obsvs, preds, obsv_lengths):
        bs = len(preds)
        self.D_optimizer.zero_grad()
        zeros = Variable(torch.zeros(bs, 1) + np.random.uniform(0, 0.05), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * np.random.uniform(0.95, 1.0), requires_grad=False).cuda()
        noise = Variable(torch.FloatTensor(torch.rand(bs, self.noise_dim)), requires_grad=False).cuda()  # uniform

        for u in range(self.improved_steps + 1):
            with torch.no_grad():
                preds_fake = self.G(obsvs, noise, obsv_lengths)
            fake_labels = self.D(obsvs, preds_fake, obsv_lengths)
            d_loss_fake = bce_loss(fake_labels, zeros)

            real_labels = self.D(obsvs, preds, obsv_lengths)  # classify real samples
            d_loss_real = bce_loss(real_labels, ones)
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()  # update D
            self.D_optimizer.step()

            if u == 0 and self.improved_steps > 0:
                backup = copy.deepcopy(self.D)

        # =============== Train Generator ================= #
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        preds_fake = self.G(obsvs, noise, obsv_lengths)

        fake_labels = self.D(obsvs, preds_fake, obsv_lengths)
        g_loss_fooling = bce_loss(fake_labels, ones)
        g_loss = g_loss_fooling

        mse_loss = F.mse_loss(preds_fake, preds)
        g_loss += 100 * mse_loss

        g_loss.backward()
        self.G_optimizer.step()

        if self.improved_steps > 0:
            self.D.load_backup(backup)
            del backup

        return g_loss.item(), d_loss.item(), mse_loss.item()

    def train(self):
        # TODO: separate train and test
        nTrain = self.n_batches * 4 // 5
        nTest = self.n_batches - nTrain

        for epoch in range(self.start_epoch, self.num_epochs):
            g_loss, d_loss, mse_loss = 0, 0, 0

            tic = process_time()
            for ii in range(nTrain):
                g_loss_ii, d_loss_ii, mse_loss_ii = self.batch_train(self.data['obsvs'][ii],
                                                                     self.data['preds'][ii],
                                                                     self.data['obsv_lengths'][ii])
                g_loss += g_loss_ii
                d_loss += d_loss_ii
                mse_loss += mse_loss_ii
            toc = process_time()
            logger.print_me(f'#{epoch:5d} | MSE = {mse_loss:.5f} | Loss G = {g_loss:.4f} '
                            f'| Loss D = {d_loss:.4f} | time = {toc - tic:.2f} s')
            if epoch % 50 == 0:  # FIXME : set the interval for running tests
                #logger.print_me(f'#{epoch:5d} | MSE = {mse_loss:.5f} | Loss G = {g_loss:.4f} '
                #                f'| Loss D = {d_loss:.4f} | time = {toc - tic:.2f} s')
                self.save(self.saving_point, epoch)

    def generate(self, starting_points, n_samples, n_step):
        past_traj = starting_points.data.unsqueeze(1).cuda()

        for step in range(1, n_step + 1):
            noise = Variable(torch.FloatTensor(torch.rand(n_samples, self.noise_dim)), requires_grad=False).cuda()
            next_step = self.G(past_traj, noise, [step] * n_samples)
            past_traj = torch.cat([past_traj, next_step.data.unsqueeze(1)], dim=1)

        return past_traj.cpu().numpy()


if __name__ == '__main__':
    # Read config file
    config_file = 'config/config.yaml'
    stream = open(config_file)
    conf = yaml.load(stream, Loader=yaml.FullLoader)

    gan = TrajectoryGAN(conf)
    #dataset, max_observation_length, obsv_lengths = gan.readData("/../TrainingData/TrajArr")
    # gan.load_model(os.path.dirname(__file__) + '/' + conf['TrajectoryGAN']['SavingPoint'])
    # gan.saving_point = os.path.dirname(__file__) + '/' + conf['TrajectoryGAN']['SavingPoint']

    training = False

    if training:
        logger = Logger(conf['Logger'])
        # Train
        gan.load_dataset("/../TrainingData/TrajArr")
        gan.load_model(conf['TrajectoryGAN']['SavingPoint'], True)
        gan.train()
    else:
        # Generate
        gan.load_model(conf['TrajectoryGAN']['SavingPoint'])
        data, steps = gan.readData("/../TrainingData/TrajArr", False)
        data = data[:20, :, :]
        n_samples = len(data)
        data_start = data[-n_samples:, 0, :]
        data = data[-n_samples:, :, :]
        #trajectories = gan.generate(torch.FloatTensor([[3, 2], [4, 3], [2, 2]]), n_samples, 5)
        trajectories = gan.generate(torch.from_numpy(data_start).float().cuda(), n_samples, steps)

        fig, ax = plt.subplots(2)
        for i in range(n_samples):
            ax[0].plot(trajectories[i, :, 0], trajectories[i, :, 1])
            ax[1].plot(data[i, :, 0], data[i, :, 1])

        ax[0].set_title('Generated')
        ax[1].set_title('Real')
        fig.show()
