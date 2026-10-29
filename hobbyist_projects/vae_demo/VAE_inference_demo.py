from PIL import Image
from model_definitions import VAEModel
import torch
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
#add mps friendliness and others


#loads initial model
model = VAEModel()
state_dict = torch.load('VAE_model.pt', weights_only=True, map_location = device)
model.to(device)
model.load_state_dict(state_dict)
model.eval()


#random VAE images
with torch.no_grad():
    z = torch.randn(16, 128)  # Sample 16 random latent vectors
    z = z.to(device)
    #print(z.shape)
    generated = model.decode(z)


for i,j in zip(generated, range(len(generated))):
  i = i.cpu()
  i = np.array(i.permute((1,2,0))*256).astype(np.uint8)
  image = Image.fromarray(i)
  image.save(f"images/img_3{j}_vae.jpg")


#VAE on CIFAR-10
from datasets import load_dataset
import numpy as np
import torch

data = load_dataset("uoft-cs/cifar10")["train"]
#data



samples = []
for c in range(0,10):
  i = data[c]["img"]
  i.save(f"images/og_image_3{c}.jpg")
  #print(torch.Tensor(np.array(i)).permute(2,0,1).shape)

  with torch.no_grad():
    image= torch.Tensor(np.array(i)).permute(2,0,1).float()/255
    image = image.to(device)
    encoded = torch.flatten(model.encoder(image))
    mu = model.fc_mu(encoded)
    logvar = model.log_var(encoded)
    z = torch.tensor(model.reparameterize(mu, logvar))
    samples.append(z)
    #print(z.shape)

samples = torch.stack(samples)
with torch.no_grad():
  images = model.decode(samples)

for c in range(0,10):
  img = images[c]
  img = img.cpu()
  img = np.array(img.permute((1,2,0))*256).astype(np.uint8)
  vae_gen = Image.fromarray(img)
  vae_gen.save(f"images/VAE_IMAGE_3{c}.jpg")
