import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=100):
        """Initialize the VAE model.

        Args:
            sigma: the standard deviation of the latent variable z, float
            obs_dim: Dimension of the observed data x, int
            latent_dim: Dimension of the latent variable z, int
            hidden_dim: Hidden dimension of the encoder/decoder networks, int
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sigma = 0
        # Trainable layers of the encoder
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear21 = nn.Linear(hidden_dim, latent_dim)
        self.linear22 = nn.Linear(hidden_dim, latent_dim)
        # Trainable layers of the decoder
        self.linear3 = nn.Linear(latent_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, obs_dim)

        # Trainable layers of the std
        self.linear5 = nn.Linear(latent_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, obs_dim)

    def encoder(self, x):
        """Obtain the parameters of q_phi(z|x) for a batch of data points.

        Args:
            x: Batch of data points, shape [batch_size, obs_dim]

        Returns:
            mu: Means of q_phi(z|x), shape [batch_size, latent_dim]
            logsigma: Log-sigmas of q_phi(z|x), shape [batch_size, latent_dim]
        """
        h = torch.relu(self.linear1(x))
        return self.linear21(h), self.linear22(h)

    def sigma_layer(self, x):
        """Obtain the log standard deviation sigma of p_theta(x|z).

        Args:
            x: Batch of data points, shape [batch_size, obs_dim]

        Returns:
            sigma: The log standard deviation of the distribution p_theta(x|z)
        """

        h = torch.relu(self.linear5(x))
        return self.linear6(h)

    def sample_with_reparam(self, mu, logsigma):
        """Draw sample from p(z) with reparametrization.

        Args:
            mu: Means of p(z) for the batch, shape [batch_size, latent_dim]
            logsigma: Log-sigmas of p(z) for the batch, shape [batch_size, latent_dim]

        Returns:
            z: Latent variables samples from p(z), shape [batch_size, latent_dim]
        """
        epsilon = torch.empty_like(mu).normal_(0., 1.)
        return epsilon * logsigma.exp() + mu

    def decoder(self, z):
        """Obtain the mean of the distribution p_theta(x|z).

        Args:
            z: Sampled latent variables, shape [batch_size, latent_dim]

        Returns:
            theta: Mean of the conditional likelihood, shape [batch_size, obs_dim]
        """
        return self.linear4(torch.relu(self.linear3(z)))


    def kl_divergence(self, mu, logsigma):
        """Compute KL divergence KL(p_theta(z|x)||p(z)).

        Args:
            mu: Means of the distributions, shape [batch_size, latent_dim]
            logsigma: Logarithm of standard deviations of the distributions,
                      shape [batch_size, latent_dim]

        Returns:
            kl: KL divergence for each of the distributions, shape [batch_size]
        """
        return 0.5 * (mu.pow(2) + (2 * logsigma).exp() - 2 * logsigma - 1).sum(-1)


    def make_prior(self):
        """Compute prior distribution. [NOT USED]

        Returns:
            The prior, a standard multivariate normal distribution.
        """

        mu = torch.zeros(self.latent_dim)
        sigma = torch.eye(self.latent_dim)
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)

    def make_posterior(self, mu, logsigma):
        """Compute posterior distribution [NOT USED]

        We draw a single sample z_i for each data point x_i.

        :args:
            mu: Means for the batch, shape [batch_size, latent_dim]
            logsigma: Log-sigmas for the batch, shape [batch_size, latent_dim]

        :Returns:
            The prosterior distributions, standard multivariate normal distributions for every mean and variance.
        """
        tensor_ = torch.diag(logsigma[0, :].exp())
        for i in range(1, self.logsigma.shape[0]):
            tensor_ = torch.cat((tensor_, torch.diag(self.logsigma[i, :].exp())), dim=0)

        return torch.distributions.multivariate_normal.MultivariateNormal(mu, tensor_.reshape((-1, 2, 2)))

    def elbo(self, x):
        """Estimate the ELBO for the batch of data.

        Args:
            x: Batch of the observations, shape [batch_size, obs_dim]

        Returns:
            elbo: Estimate of ELBO for each sample in the batch, shape [batch_size]
        """
        mu, logsigma = self.encoder(x)
        z = self.sample_with_reparam(mu, logsigma)
        theta = self.decoder(z)
        self.sigma = self.sigma_layer(z)
        samples = self.sample_with_reparam(theta, self.sigma)
        # log_obs_prob = torch.distributions.normal.Normal(theta, sigma.exp()).log_prob(x).sum(-1)
        # Computing the mse directly speeds up the training
        log_obs_prob = -F.mse_loss(samples, x, reduction='none').sum(-1)
        kl = self.kl_divergence(mu, logsigma)
        return log_obs_prob - kl

    def sample(self, num_samples):
        """Generate samples from the model.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            x: Samples generated by the model, shape [num_samples, obs_dim]
        """
        z = torch.empty(num_samples, self.latent_dim).normal_()
        theta = self.decoder(z)
        sigma = self.sigma_layer(z)
        samples = self.sample_with_reparam(theta, sigma)
        return samples



