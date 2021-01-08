import numpy as np
import torch
import torch.nn as nn
import utils
from utils import *


def train_MNIST(vae, train_loader, test_loader, obs_dim, max_epochs=100, display_step=50):
    vae.train()
    train_loss_avg = []
    max_epochs = max_epochs
    epoch_count = []
    display_step = display_step
    lr = 1e-3
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    epochprints = [0,1, 5, 25, 50, max_epochs]
    for epoch in range(max_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        print(f'Epoch {epoch}')
        for ix, batch in enumerate(train_loader):
            x, y = batch
            x = x.view(x.shape[0], obs_dim)
            opt.zero_grad()
            loss = -vae.elbo(x).mean(-1)
            loss.backward()
            opt.step()
            if ix % display_step == 0:
                print(f'  loss = {loss.item():.2f}')
            num_batches += 1
            # Calculate loss on the test set
            images, labels = iter(test_loader).next()
            images = images.view(images.shape[0], obs_dim) 
            loss = -vae.elbo(images).mean(-1)
            train_loss_avg[-1] += loss.item()


        train_loss_avg[-1] /= num_batches
        epoch_count.append(epoch)

        if epoch in epochprints:
            # Plot latent space
            utils.plot_latent_space(x,y,vae)
            images, labels = iter(test_loader).next()
            # Plot samples
            utils.visualize_samples(images[0:20].view(-1, 28, 28))
            # Plot reconstructed images
            print("Reconstructed images")
            utils.reconstruct_output(images[0:20], vae, obs_dim)
            utils.visualize_samples(images[0:20].view(-1, 28, 28))
            print("VAE samples:")
            # Plot generated samples
            x = vae.sample(20).view(-1, 28, 28).detach().cpu().numpy()
            utils.visualize_samples(x)


def train_FireEvacuation(vae, train_loader, test_loader, obs_dim, max_epochs=200, display_step=100):
    max_epochs = max_epochs
    display_step = display_step
    lr = 5e-4
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')
        for ix, batch in enumerate(train_loader):
            x = batch
            x = x.view(x.shape[0], obs_dim)
            opt.zero_grad()
            loss = -vae.elbo(x).mean(-1)
            loss.backward()
            opt.step()
            for _, batch_ in enumerate(test_loader):
                loss_ = -vae.elbo(batch_).mean(-1)

            if ix % display_step == 0:
                print(f'  loss = {loss_.item():.9f}')