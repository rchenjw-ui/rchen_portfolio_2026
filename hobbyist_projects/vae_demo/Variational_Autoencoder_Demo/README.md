# Variational_Autoencoder_Demo
A demonstration of variational autoencoders on CIFAR-10, using convolutional neural networks for encoder and deconvolution for encoder

# Learnings
- Initially, the model was far too small and trained on far too little data. Bitter lesson of deep learning again! 
- I finally understand the random noise injection stuff, It's basically a way to A, force the VAE to generalize a bit better, and B, effectively forces the VAE to learn a whole distribution, and C, allows it to work on noise it hasn't seen
- The reparameterization trick makes a lot more sense: It goes from random sampling (not differentiable) to adding noise multiplied by random constant (which is differentiable)
