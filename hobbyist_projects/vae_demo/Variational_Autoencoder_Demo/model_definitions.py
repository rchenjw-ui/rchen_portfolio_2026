import torch
import torch.nn as nn
import numpy as np


#define encoder and decoder

import torch.nn as nn
import torch.nn.init as init


class VAEModel(nn.Module):

  def __init__(self):
    super().__init__()

    """
    Original design, revised as far too small 
    
    self.encoder = nn.Sequential(
    #cnn and other layers
    nn.Conv2d(3,6,3),
    nn.MaxPool2d(2),
    nn.Conv2d(6,1,5),
    nn.Flatten(),
    nn.Linear(121,64),
    nn.LeakyReLU()
    )
    self.fc_mu = nn.Linear(64, 128)
    self.log_var = nn.Linear(64, 128)
    self.decoder = nn.Sequential(
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 128),
    nn.LeakyReLU(),
    nn.Unflatten(1, (8, 4, 4)),  # reshape to (8, 4, 4)
    nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1),  # → 16x8x8
    nn.LeakyReLU(),
    nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # → 3x16x16
    nn.Upsample(scale_factor=2), # → 3x32x32
    nn.Sigmoid()
    """
    self.encoder = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),   # 32x32x3 → 32x32x32
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 16x16x64
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, stride=2, padding=1), # → 8x8x128
    nn.ReLU(),
    nn.Conv2d(128, 256, 3, stride=2, padding=1), # → 4x4x256
    nn.ReLU(),
    nn.Flatten(),  # → 4096 features
)
    self.fc_mu = nn.Linear(4096, 128)
    self.log_var = nn.Linear(4096, 128)


    self.decoder = nn.Sequential(
    nn.Linear(128, 4096),
    nn.ReLU(),
    nn.Unflatten(1, (256, 4, 4)),
    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # → 8x8x128
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # → 16x16x64
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # → 32x32x32
    nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),  # → 32x32x3
    nn.Sigmoid()  # Output [0,1] to match normalized input
)

  def encode(self, x):
    logits = self.encoder(x)
    mu = self.fc_mu(logits)
    log_var = self.log_var(logits)
    return mu, log_var

  def reparameterize(self, mu, log_var):
    log_var = torch.clamp(log_var, min=-5, max=5)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps

  def decode(self, z):
    decoded = self.decoder(z)
    return decoded

  def forward(self, x):
    mu, log_var = self.encode(x)
    z = self.reparameterize(mu, log_var)
    return self.decode(z)



  #testing code
"""
vae = VAEModel()

data = []
for c in range(0,5):
  data.append(torch.Tensor(np.zeros((3,32,32))))

data = torch.tensor(np.array(data))
out = vae.encode(data)
print(out[0].shape)
print(out[1].shape)

def loss(outputs, means,log_vars, samples):
  MSE_loss = nn.MSELoss()
  XEL = MSE_loss(outputs, samples)
  kl_divergence = -0.5 * torch.sum(1+log_vars - means*means - (torch.exp(log_vars)))
  return kl_divergence + XEL

"kL divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)"
z = vae.reparameterize(out[0], out[1])
print(loss(i, out[0], out[1], data))
"""
