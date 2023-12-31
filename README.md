# DeepLearningPortfolio
DeepLearningPortfolio is a comprehensive collection of Jupyter notebooks showcasing my deep learning projects and experiments.

## 1. Softmax Loss
The "Softmax Loss" directory in this repository contains two Jupyter notebooks that focus on the implementation and application of the softmax loss function in neural networks. Here's a brief overview of each notebook:

### 1.1. softmax.ipynb (Softmax Loss and SGD)
This notebook is dedicated to the implementation and experimentation with the softmax loss function. Key highlights include:

- **Detailed Implementation:** Step-by-step implementation of the softmax loss function, ensuring a clear understanding of its mechanics.
- **Numerical Stability Considerations:** Addresses common numerical issues associated with softmax computations.
- **Stochastic Gradient Descent (SGD):** Integration of the softmax loss function with SGD, demonstrating how it is used for optimizing a model.
- **Testing and Validation:** The notebook includes various tests to validate the correctness and efficiency of the implemented softmax loss function.

### 1.2. two_layer_net.ipynb (Two-Layer MLP Network with Softmax Loss)
This notebook expands on the softmax loss function by integrating it into a two-layer Multi-Layer Perceptron (MLP) network. It includes:

- **Network Architecture:** Implementation of a two-layer fully-connected neural network with ReLU activation.
- **Softmax Loss Integration:** Utilization of the softmax loss function from the previous notebook for network training.
- **Hyperparameter Tuning:** Exploration of different hyperparameters to optimize the network's performance.
- **Visualization and Analysis:** The notebook provides visualizations and in-depth analysis of the network's learning process.

## 2. Convolutional Neural Networks (CNNs, ConvNets)

The "Convolutional Neural Networks (CNNs, ConvNets)" directory in this repository focuses on the implementation and application of CNNs using the PyTorch framework. This directory is an essential resource for understanding the fundamentals and advanced concepts of CNNs. It contains the following Jupyter notebook:

### 2.1. PyTorch.ipynb

This notebook is a comprehensive exploration of CNNs using PyTorch, one of the most popular deep learning frameworks. The key aspects covered in this notebook include:

- **Introduction to CNN Architecture:**
Detailed explanation of the CNN architecture, including convolution layers, pooling layers, and fully-connected layers.

- **PyTorch Framework Usage:**
Step-by-step guide on building CNN models in PyTorch, showcasing the framework's flexibility and ease of use.

- **Image Classification Tasks:**
Implementation of CNNs for image classification tasks, demonstrating the practical application of these networks.

- **Experimentation with CNN Layers:**
Exploration of various layers and techniques used in CNNs, such as different types of convolution and normalization techniques.

- **Hyperparameter Tuning:**
Insights into tuning hyperparameters to enhance model performance, accompanied by practical examples.

- **Visualizations:**
Includes visualizations of training progress, feature maps, and model predictions to offer a clear understanding of how CNNs process and interpret image data.

## 3. Variational Autoencoder (VAE)

The "Variational Autoencoder (VAE)" directory of this repository is dedicated to the exploration and implementation of Variational Autoencoders, a class of generative models. This directory includes the following Jupyter notebook:

### 3.1. VAE.ipynb

This notebook serves as a deep dive into the world of Variational Autoencoders, utilizing them for generative tasks. Key features of this notebook include:

- **Fundamentals of VAEs:**
A comprehensive introduction to the concepts and mathematical foundations underlying Variational Autoencoders.

- **Implementation in Python:**
Step-by-step implementation of a VAE, showcasing how these models are built from scratch.

- **Training and Testing:**
Detailed explanation of the training process, including the optimization of the variational lower bound.

- **Generative Modeling:**
Demonstrations of VAEs in generating new data, highlighting their capability as generative models.

- **Visualization of Results:**
Includes visualizations of the latent space and the generated samples to provide insights into the model's performance and capabilities.

- **Applications:**
Exploration of various applications of VAEs, such as image generation and reconstruction, showcasing their versatility.

## 4. Generative Adversarial Networks (GANs)

This directory is dedicated to exploring Generative Adversarial Networks (GANs), a groundbreaking and widely popular approach in generative modeling. The directory contains the following Jupyter notebook:

### 4.1. Generative_Adversarial_Networks_TF.ipynb
This notebook offers a comprehensive guide to understanding and implementing GANs using TensorFlow. It is designed to provide both theoretical insights and practical coding experience. Key highlights of the notebook include:

- A thorough introduction to the GAN architecture and its working principles.
Detailed TensorFlow implementation of a GAN model.
- Exploration of the training dynamics and challenges in training GANs.
- Visualization of generated outputs to demonstrate the effectiveness of the model.
- Discussion on various applications of GANs in different domains.

## 5. Adversarial Attacks (White Box Attack and Black Box Attack)

*to be updated*

## 6. KL Divergence

*to be updated*

## 7. Reinforcement Learning

This directory focuses on Reinforcement Learning (RL), a crucial area of machine learning where agents learn to make decisions by interacting with an environment. It includes two comprehensive Jupyter notebooks demonstrating key RL algorithms applied to different simulation environments.

### 7.1. Vanilla Policy Gradient (VPG) for Cartpole
- **Filename:** `cartpole_vpg.ipynb`
- **Description:**
- This notebook provides an implementation of the Vanilla Policy Gradient (VPG) algorithm, applied to the classic Cartpole balancing problem.
- Key Learning Outcomes:
    - Understanding the fundamentals of policy-based reinforcement learning.
    - Hands-on experience with implementing VPG using PyTorch.
    - Visualization and analysis of the agent's performance in maintaining the pole's balance.

### 7.2. Deep Q-Network (DQN) for Lunar Lander
- **Filename:** `lunarlander_dqn.ipynb`
- **Description:**
- Explore the Deep Q-Network (DQN) algorithm, a value-based method in RL, applied to the Lunar Lander environment from OpenAI Gym.
- Key Learning Outcomes:
    - Detailed walkthrough of setting up and training a DQN.
    - Strategies for optimizing the learning process.
    - Insights into the agent's decision-making process in a more complex environment compared to Cartpole.

## 8. Mixup (Vanilla classifier, Input mixup, Manifold mixup, CutMix)

*to be updated*

## 9. Continuous Bag of Words (CBOW) Model

This directory contains a Jupyter notebook that delves into the Continuous Bag of Words (CBOW) model, a fundamental concept in natural language processing (NLP). In NLP, CBOW is commonly employed to predict a word based on context, making it a key methodology in areas such as word embeddings and language modeling. 

### 9.1. CBOW.ipynb
- This notebook provides a comprehensive walkthrough of the CBOW model. 
- It includes:
  - Introduction to the CBOW model's theory and architecture.
  - Step-by-step implementation of the model.
  - Techniques for training and evaluating the model on text data.
  - Exploration of word embeddings generated by the model.
- Key Learning Outcomes:
    - Grasping the mechanics of word embeddings and their significance in NLP.
    - Gaining hands-on experience in training a CBOW model.
    - Understanding the practical applications of the CBOW model in various NLP tasks.