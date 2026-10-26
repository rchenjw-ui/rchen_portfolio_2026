# rchen_portfolio
MIT Maker Portfolio (code)

Files all located in hobbyist_projects
Folders:
- chla: the tumor fraction estimation project (bioinformatics). 
  - Using DNA methylation in cell-free DNA from aqueous humor, estimates retinoblastoma tumor fraction. 
  - Uses a logistic-regression based tumor-fraction classifier
- rbm_mnist: demonstrates two deep-learning algorithms: Backpropagation vs Deep Belief Network/Restricted Boltzmann Machine. 
  - Both are used to train a equivalently-sized neural network on MNIST dataset. 
  - In addition, as deep belief networks are trained layer-by-layer, trains a small machine-learning model on each layer to see accuracy
- homework: CS50AI's homework folder. 
  - Contains several data structures/algorithms heavy problem sets. All unit tests were passed 
- vae_demo: Demonstrates a variational autoencoder on CIFAR-10. 
  - Chronologically the first one, the first time I implemented an ML algorithm on my own  
- cheminformatics: Demo comparison of featurization methods for Tox21 ML classifier. 
  - Turns out, the split method is quite important! 
  - Splitting via structure produced significantly worse results than a purely random split
  - The implication is generalization to novel structures is not always testable via classical train/test splits 
- chatbot_rag: A chatbot (via Llama 3.2 3B) with encrypted chat history and retrieval-augmented generation 
  - (encryption is important: today it's HuggingFace, tomorrow it could be Google Drive)
- jev_classifier: Testing open-source Jev-type "System One" models as a classification/retrieval system

Note: The chatbot_rag file requires two .gguf files: one for the LLM and one for the retrieval model. 
I used Llama 3.2 3B and ModernBert-Embed_Base, but you can pick any model. In addition, more will be added.
