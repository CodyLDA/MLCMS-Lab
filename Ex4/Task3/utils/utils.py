import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F


def plot_latent_space(x,y, vae):
    latent = vae.encoder(x)[0]
    latent_embedded = TSNE(init='pca', n_components=2).fit_transform(latent.detach().numpy())
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(
        latent_embedded[:, 0],
        latent_embedded[:, 1],
        c=y,
       marker='o', edgecolor='none')
    plt.colorbar(ticks=range(N))
    plt.grid(True)
    plt.show()


def reconstruct(images, vae, obs_dim):
    vae.eval()
    with torch.no_grad():
            latent_mu, latent_logvar = vae.encoder(images.view(images.shape[0], obs_dim))
            latent = vae.sample_with_reparam(latent_mu, latent_logvar)
            theta = vae.decoder(latent)
            sigma = vae.sigma_layer(theta)
            samples = vae.sample_with_reparam(theta, sigma)
            samples = samples.detach().cpu().numpy()
            return samples


def reconstruct_output(images, vae, obs_dim):
    reconstruction = reconstruct(images, vae, obs_dim)
    return reconstruction


def visualize_samples(samples, num_rows=4, num_cols=5):

    sns.set_style('white')
    num_total = num_rows * num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 2 * num_rows))
    for i in range(num_total):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(samples[i], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'Sample #{i+1}')
    plt.tight_layout()
    plt.show()


def visualise_output(vae, images=None, processed=True, num_samples = 0):
    if processed:
        return images
    else:
        images = process_data(images, num_samples)
        with torch.no_grad():
            for ix, batch in enumerate(images):
                latent_mu, latent_logvar = vae.encoder(batch)
                latent = vae.sample_with_reparam(latent_mu, latent_logvar)
                theta = vae.decoder(latent)
                sigma = vae.sigma_layer(theta)
                samples = vae.sample_with_reparam(theta, sigma)
                samples = samples.detach().cpu().numpy()
        return samples


def process_data(images, num_samples):
    data = read_data(num_samples)
    indexes = np.random.randint(low=0, high=data.shape[0], size=num_samples)
    dist_new = np.ones((num_samples,2)).astype(np.float32)
    sample = 0
    for index in indexes:
        dist_new[sample, :] = data[index].astype(np.float32)
        sample += 1
    test_loader = torch.utils.data.DataLoader(
        dist_new,
        batch_size=num_samples, shuffle=True
    )
    return test_loader


def read_data(num_samples):
    train_data = np.load('FireEvac_train_set.npy')
    test_data = np.load('FireEvac_test_set.npy')
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    data = np.concatenate((test_data, train_data))
    data += np.random.randn(data.shape[0], 2).astype(np.float32)
    return data


def sample(num_samples, vae):
    # Samples data for the generation of new samples
    # Since the learnt distribution does not seem to match the test/train set distributions
    # This is done by reconstructing noisy data points
    # (Only for the FireEvacuation model)
    if vae.latent_dim != 2:
        print("Wrong latent dimensions, please make sure it is set to 2")
        return 1
    else:
        return visualise_output(vae, processed=False, num_samples=num_samples)
