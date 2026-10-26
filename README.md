# rchen_portfolio
MIT Maker Portfolio (code)

Folders:
- chla: the tumor fraction estimation project (bioinformatics).
  - Using DNA methylation in cell-free DNA from aqueous humor, estimates retinoblastoma tumor fraction.
  - Uses a logistic-regression based tumor-fraction classifier
- rbm_mnist: demonstrates two deep-learning algorithms: Backpropagation vs Deep Belief Network/Restricted Boltzmann Machine.
  -Both are used to train a equivalently-sized neural network on MNIST dataset.
  In addition, as deep belief networks are trained layer-by-layer, trains a small machine-learning model on each layer to see accuracy
- CS50AI: CS50AI's homework folder.
  - Contains several data structures/algorithms heavy problem sets. All unit tests were passed 
- vae_demo: Demonstrates a variational autoencoder on CIFAR-10.
  - Chronologically the first one, the first time I implemented an ML algorithm on my own  
- cheminformatics: Demo comparison of featurization methods for Tox21 ML classifier.
  - Turns out, the split method is quite important!
  - Splitting via structure produced significantly worse results than a purely random split
  - the implication is generalization to novel structures is not always testable via classical train/test splits 
- chatbot_rag: A chatbot (via Llama 3.2 3B) with encrypted chat history and retrieval-augmented generation
  - (encryption is important: today it's HuggingFace, tomorrow it could be Google Drive)
- jev_classifier: Testing open-source Jev-type "System One" models as a classification/retrieval system
