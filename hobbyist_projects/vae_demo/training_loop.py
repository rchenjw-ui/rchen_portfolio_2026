"""
Notes about VAEs:

- encoder returns logits
- the logits go through 2 linear
- leads to noise and mean

- reparameterization uses a random multiplier

- decoder uses unflatten, conv transpose and up sampling

kL divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
"""

from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from model_definitions import VAEModel

data = load_dataset("uoft-cs/cifar10")["train"]


def train(VAE, dataset, loss_fn, optim, epochs, batch_size, device):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch}...")

        #Shuffle and batch properly

        indices = torch.randperm(len(dataset))

        for batch_start in range(0, len(dataset), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Get batch
            samples = []
            for idx in batch_indices:
                img = torch.tensor(np.array(dataset[int(idx)]["img"])).permute(2, 0, 1).float() / 255.0
                samples.append(img)
            samples = torch.stack(samples)
            samples = samples.to(device)

            # Forward pass
            optim.zero_grad()
            mu, log_var = VAE.encode(samples)
            z = VAE.reparameterize(mu, log_var)
            recon = VAE.decode(z)

            # Loss and backward
            batch_loss = loss_fn(recon, mu, log_var, samples)
            batch_loss.backward()
            optim.step()

            if batch_start % 1000 == 0:
                print(f"  Batch {batch_start}, Loss: {batch_loss.item():.4f}")
    return model


def loss(outputs, means, log_vars, samples):
    # Use MSE for reconstruction
    recon_loss = nn.functional.mse_loss(outputs, samples, reduction='sum')

    # KL divergence (your formula is correct)
    kl_divergence = -0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())

    # Balance the terms (typical β-VAE)
    beta = 0.01  # Start low!
    return recon_loss + beta * kl_divergence

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

model = VAEModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=0.005)

model = train(model, data,loss, optimizer, 5, 100, device)






#permute to (1,2,0) when done