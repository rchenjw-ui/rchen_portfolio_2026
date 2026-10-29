# vae_demo — Variational Autoencoder on CIFAR-10

First full ML algorithm implemented end-to-end in this portfolio: a **convolutional VAE** trained on CIFAR-10 (32×32 RGB).

Encoder: strided conv stack → latent \(\mu, \log\sigma^2\) (dim 128).  
Decoder: linear expand → conv-transpose stack → sigmoid RGB reconstruction.

## Learnings

# Variational_Autoencoder_Demo
A demonstration of variational autoencoders on CIFAR-10, using convolutional neural networks for encoder and deconvolution for encoder

# Learnings
- The first architecture was far too small and under-trained — a direct reminder of the “bitter lesson.”
- In addition, changing loss
- Reparameterization: sampling \(z = \mu + \sigma \odot \varepsilon\) turns a non-differentiable draw into a differentiable path so gradients reach the encoder.
- Injecting noise forces the model to cover a distribution, not a single point estimate, which is what makes generation and interpolation possible.

## Requirements

```bash
pip install torch torchvision pillow numpy
# CUDA optional; CPU works for inference on the saved weights
```

## Files

| File | Role |
|------|------|
| `model_definitions.py` | `VAEModel` (encoder / reparameterize / decoder) |
| `VAE_model.pt` | Trained weights |
| `VAE_inference_demo.py` | Sample random latents + reconstruct a few CIFAR-style images |
| `training_loop.py` | Similar inference / export script (historical) |

## Run inference

```bash
cd vae_demo
python VAE_inference_demo.py
```

This writes:

- `img_3*_vae.jpg` — samples from \(\mathcal{N}(0,I)\) through the decoder  
- `og_image_3*.jpg` / `VAE_IMAGE_3*.jpg` — original vs reconstruction pairs (when CIFAR samples are available in the runtime)

If you do not have a GPU:

```bash
# scripts assume .cuda(); for CPU, change device to "cpu" or set:
export CUDA_VISIBLE_DEVICES=""
# and edit the two `.cuda()` calls to `.to("cpu")` if needed
```

## Architecture (current)

```text
Encoder:  3×32×32
  → Conv 32 → 64 → 128 → 256 (stride-2)
  → Flatten 4096 → μ, log σ²  (128-d)

Decoder:  z (128)
  → Linear 4096 → 256×4×4
  → ConvTranspose → 32×32×3 + Sigmoid
```

Loss used in training (see comments in `model_definitions.py`): reconstruction (MSE) + KL to standard normal.

## Notes

- Chronologically the earliest project in the set; kept as a record of the first from-scratch generative model.
- Weights are included (`VAE_model.pt`) so the demo runs without re-training.
- For a clean portfolio run, prefer `MPLBACKEND=Agg` if you add any matplotlib later; current scripts only write JPEGs via Pillow.
