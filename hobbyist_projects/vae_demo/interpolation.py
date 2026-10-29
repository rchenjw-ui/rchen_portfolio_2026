"""
Latent interpolation between two random CIFAR-10 images.
Saves interpolated_image_1.jpg ... interpolated_image_10.jpg
"""

from datasets import load_dataset
import numpy as np
import torch
from PIL import Image
from model_definitions import VAEModel

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# load model
model = VAEModel()
state_dict = torch.load("VAE_model.pt", weights_only = True, map_location = device)
model.load_state_dict(state_dict)
model.eval()


model = model.to(device)

data = load_dataset("uoft-cs/cifar10")["train"]

# two random CIFAR images
idx_a = int(torch.randint(0, len(data), (1,)).item())
idx_b = int(torch.randint(0, len(data), (1,)).item())

img_a = torch.tensor(np.array(data[idx_a]["img"])).permute(2, 0, 1).float() / 255.0
img_b = torch.tensor(np.array(data[idx_b]["img"])).permute(2, 0, 1).float() / 255.0

# also save the originals
Image.fromarray(np.array(data[idx_a]["img"])).save("interp_source_a.jpg")
Image.fromarray(np.array(data[idx_b]["img"])).save("interp_source_b.jpg")

img_a = img_a.unsqueeze(0).to(device)
img_b = img_b.unsqueeze(0).to(device)

with torch.no_grad():
    mu_a, log_var_a = model.encode(img_a)
    mu_b, log_var_b = model.encode(img_b)
    # use means for a clean path (could also reparameterize)
    z_a = mu_a
    z_b = mu_b

# 10 steps between a and b
for i in range(10):
    t = i / 9.0
    z = (1.0 - t) * z_a + t * z_b
    with torch.no_grad():
        out = model.decode(z)
    out = out.squeeze(0).cpu()
    arr = np.array(out.permute(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(arr)
    image.save(f"interpolated_image_{i + 1}.jpg")
    print(f"wrote interpolated_image_{i + 1}.jpg")

print("done")