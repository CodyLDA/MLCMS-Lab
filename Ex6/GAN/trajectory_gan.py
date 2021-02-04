import os
import yaml
import glob
from time import process_time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.nn.utils.rnn import pad_sequence
from util.helper import bce_loss
from util.debug_utils import Logger
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator


class TrajectoryGAN:
    def __init__(self, config):
        """
        Initializing the Trajectory GAN for generating trajectories

        Parameters
        ----------
        config : yaml Fileloader for traversing config settings
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # some minor adjustments
        self.base_distr_dim = config['TrajectoryGAN']['BaseDistrDim']
        self.num_epochs = config['TrajectoryGAN']['NumEpoch']
        self.batch_size = config['TrajectoryGAN']['BatchSize']

        # The hyperparameters
        self.improved_steps = config['TrajectoryGAN']['ImprovedDiscriminator']  # how far the discriminator is ahead

        # generator hyperparameters
        generator_fc_hidden = config['TrajectoryGAN']['GeneratorHidden']
        generator_lstm_hidden = config['TrajectoryGAN']['GeneratorLSTM']
        generator_learning_rate = config['TrajectoryGAN']['GeneratorLearning']

        # discriminator hyperparameters
        discriminator_fc_hidden = config['TrajectoryGAN']['DiscriminatorHidden']
        discriminator_lstm_hidden = config['TrajectoryGAN']['DiscriminatorLSTM']
        discriminator_learning_rate = config['TrajectoryGAN']['DiscriminatorLearning']

        # shared parameters between generator and discriminator
        embedding_config = config['TrajectoryGAN']['Embedding']
        leaky_relu_slope = config['TrajectoryGAN']['LeakyRelu']

        # optimizer parameters
        beta_1 = config['TrajectoryGAN']['Beta1']
        beta_2 = config['TrajectoryGAN']['Beta2']

        # setting up the architecture objects
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
        """
        Reading the data from the path and splitting for training or testing
        Returning the data and the longest trajectory, both used for preprocessing the data
        in the load_dataset() method

        Parameters
        ----------
        dataset_path: string, path to the data
        training_data: boolean, True -> reading training data, False -> reading test data

        Returns
        -------
        loaded_data: (num_samples, trajectory_length, 2) ndarray,
            num_samples -> number of trajectories encountered
            trajectory_length -> length of a trajectory enlarged to the maximum length if shorter
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at a time step
        max_traj_length: int, the maximal trajectory length encountered in the data
        """
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
        """
        Setting up the data used and preprocessing for training
        Data originally contains whole trajectories
        Splitting the trajectories into sub-trajectories up to each time step
        Further setting up a prediction target for each sub-trajectory

        Parameters
        ----------
        dataset_path: string, path to the data

        Returns
        -------
        """
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
            # original trajectory split into sub-trajectories and stacked
            self.data['real_traj'].append(step_by_step_traj[batch_index * batch_size:
                                                            min((batch_index + 1) * batch_size, self.num_data)])
            # length information for each sub-trajectory
            self.data['traj_lengths'].append(traj_lengths[batch_index * batch_size:
                                                          min((batch_index + 1) * batch_size, self.num_data)])
            # prediction targets for each sub-trajectories
            self.data['prediction'].append(next_step_predictions[batch_index * batch_size:
                                                                 min((batch_index + 1) * batch_size, self.num_data)])

    def load_model(self, saving_path):
        """
        Loading a trained model or a model that should be continued to be trained

        Parameters
        ----------
        saving_path: string, path to a trained model

        Returns
        -------
        """
        self.start_epoch = 1
        # only loads a proper model
        assert os.path.isfile(saving_path), f'No model under {saving_path} found!'
        print('Load model from ' + saving_path)
        save_point = torch.load(saving_path, map_location=self.device)
        self.start_epoch = save_point['epoch'] + 1  # set new epoch if used for further training
        self.G.load_state_dict(save_point['G_dict'])
        self.D.load_state_dict(save_point['D_dict'])
        self.G_optimizer.load_state_dict(save_point['G_optimizer'])
        self.D_optimizer.load_state_dict(save_point['D_optimizer'])

    def save_model(self, saving_path, model_name, epoch):
        """
        Saving a model to a given location with given name attached with the epoch when it was saved

        Parameters
        ----------
        saving_path: string, path to save the model
        model_name: string, name of the model that was trained
        epoch: int, epoch to which the model was trained so far

        Returns
        -------
        """
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
        """
        Training per batch to update the discriminator and the generator

        Parameters
        ----------
        real_traj: (batch_size, num_subtraj, 2) tensor, the original sub-trajectories
            batch_size -> number of samples encountered in one batch
            num_subtraj -> the number of sub-trajectories created from one trajectory
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at a time step
        prediction: (batch_size, 2) tensor, the prediction targets for each sub-trajectory
            batch_size -> number of samples encountered in one batch
            2 -> last dimension pertains the x- and y-coordinate of the trajectory at a time step
                here this only contains the prediction targets for the sub-trajectories
        traj_lengths: list of length batch_size containing the length of the sub-trajectories

        Returns
        -------
        g_loss: float, generator loss value
        d_loss: float, discriminator loss value
        mse_loss: float, mse between the generated and original data
        """
        batch_size = len(prediction)

        zeros = torch.zeros(batch_size, 1).to(self.device)  # reference for fakes trajectories
        ones = torch.ones(batch_size, 1).to(self.device)  # reference for real trajectories
        base_distr = torch.rand(batch_size, self.base_distr_dim).float().to(self.device)  # uniform base distribution
        zeros.requires_grad, ones.requires_grad, base_distr.requires_grad = False, False, False

        # train the discriminator first for the amount of improvement steps wanted
        self.D_optimizer.zero_grad()
        for improvement in range(self.improved_steps + 1):
            with torch.no_grad():
                prediction_fake = self.G(real_traj, base_distr, traj_lengths)  # generate the fake next step

            fake_labels = self.D(real_traj, prediction_fake, traj_lengths)  # discriminate the predictions
            d_loss = bce_loss(fake_labels, zeros)  # setting the loss if discriminator thought the fakes were real

            real_labels = self.D(real_traj, prediction, traj_lengths)  # discriminate the real prediction labels
            d_loss += bce_loss(real_labels, ones)  # adding discriminator loss if thought the real were fake

            d_loss.backward()
            self.D_optimizer.step()

        # train the generator with a improved version of the discriminator
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        prediction_fake = self.G(real_traj, base_distr, traj_lengths)  # generate the fake next step

        fake_labels = self.D(real_traj, prediction_fake, traj_lengths)  # discriminate the predictions
        g_loss = bce_loss(fake_labels, ones)  # setting generator loss if was not able to fool the discriminator
        mse_loss = 100 * F.mse_loss(prediction_fake, prediction)  # also add loss if differs from original
        g_loss += mse_loss

        g_loss.backward()
        self.G_optimizer.step()

        return g_loss.item(), d_loss.item(), mse_loss.item()

    def train(self, saving_path, model_name):
        """
        Begin training of a model given a saving path for the model and a model name under which name
        the model should be saved

        Parameters
        ----------
        saving_path: string, path to save the model
        model_name: string, name of the model that was trained

        Returns
        -------
        """
        for epoch in range(self.start_epoch, self.num_epochs):
            g_loss, d_loss, mse_loss = 0, 0, 0

            tic = process_time()  # measuring time for each epoch needed during training
            for index in range(self.num_batches):
                g_loss_index, d_loss_index, mse_loss_index = self.train_batch(self.data['real_traj'][index],
                                                                              self.data['prediction'][index],
                                                                              self.data['traj_lengths'][index])
                g_loss += g_loss_index
                d_loss += d_loss_index
                mse_loss += mse_loss_index
            toc = process_time()

            # logging the process of an epoch with the MSE, generator and discriminator loss as well as time needed
            logger.print_me(f'#{epoch:5d} | MSE = {mse_loss:.5f} | Loss G = {g_loss:.4f} '
                            f'| Loss D = {d_loss:.4f} | time = {toc - tic:.2f} s')

            if epoch % 50 == 0:  # save a model for each 50 epochs
                self.save_model(saving_path, model_name, epoch)

    def generate(self, starting_points, num_step):
        """
        Generate new trajectories iteratively from given starting points for a given number of steps

        Parameters
        ----------
        starting_points: (num_samples, 2) tensor, the initial starting points of the trajectories
            num_samples: number of trajectories to be generated
            2 -> last dimension pertains the x- and y-coordinate of the trajectory initially
        num_step: int, number of time steps to be generated

        Returns
        -------
        past_traj: (num_samples, num_step, 2) ndarray, the trajectories generated
            num_samples: number of trajectories to be generated
            num_step: length of the generated trajectory
            2 -> last dimension pertains the x- and y-coordinate of the trajectory initially
        """
        past_traj = starting_points.data.unsqueeze(1).to(self.device)
        num_samples = starting_points.shape[0]

        for step in range(1, num_step + 1):
            base_distr = torch.rand(num_samples, self.base_distr_dim).float().to(self.device)
            base_distr.requires_grad = False

            next_step = self.G(past_traj, base_distr, [step] * num_samples)
            past_traj = torch.cat([past_traj, next_step.data.unsqueeze(1)], dim=1)

        print(f'Generated {num_samples} new trajectories of length {num_step}.')
        return past_traj.cpu().numpy()


# used for training and debug, was not exported into other file
# since unnecessary in this case
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
