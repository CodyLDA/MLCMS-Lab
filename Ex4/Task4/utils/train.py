import numpy as np
import torch
import torch.nn as nn
import utils
from utils import *


def train_MNIST(vae, train_loader, test_loader, obs_dim, max_epochs=100, display_step=10):
    vae.train()
    train_loss_avg = []
    max_epochs = max_epochs
    epoch_count = []
    display_step = display_step
    lr = 5e-4
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    epochprints = [1, 5, 25, 50, max_epochs]
    for epoch in range(max_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        print(f'Epoch {epoch}')
        for ix, batch in enumerate(train_loader):
            x, y = batch
            x = x.view(x.shape[0], obs_dim) / 255
            opt.zero_grad()
            loss = -vae.elbo(x).mean(-1)
            loss.backward()
            opt.step()
            num_batches += 1
            images, labels = iter(test_loader).next()
            images = images.view(images.shape[0], obs_dim) / 255
            opt.zero_grad()
            loss = -vae.elbo(images).mean(-1)
            train_loss_avg[-1] += loss.item()

        train_loss_avg[-1] /= num_batches
        epoch_count.append(epoch)
        if epoch % display_step == 0:
            print('  loss =' + str(train_loss_avg[-1]))
        if epoch in epochprints:
            plot_latent_space(x,y,vae)
            images, labels = iter(test_loader).next()
            visualize_samples(images[0:20].view(-1, 28, 28))
            print("Reconstructed images")
            reconstruct_output(images[0:20], vae)
            visualize_samples(images[0:20].view(-1, 28, 28))
            print("VAE samples:")
            x = vae.sample(20).view(-1, 28, 28).detach().cpu().numpy()
            visualize_samples(x)


def train_FireEvacuation(vae, train_loader, test_loader, obs_dim, max_epochs=200, display_step=100):
    max_epochs = max_epochs
    display_step = display_step
    lr = 5e-4
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')
        for ix, batch in enumerate(train_loader):
            x = batch
            x = x.view(x.shape[0], obs_dim)  # we flatten the image into 1D array
            opt.zero_grad()
            # We want to maximize the ELBO, so we minimize the negative ELBO
            loss = -vae.elbo(x).mean(-1)
            loss.backward()
            opt.step()
            for _, batch_ in enumerate(test_loader):
                loss_ = -vae.elbo(batch_).mean(-1)

            if ix % display_step == 0:
                print(f'  loss = {loss_.item():.9f}')