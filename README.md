# TEQUAL

<u>T</u>oplogical <u>Eq</u>uivalency Classes for <u>U</u>nderstanding <u>A</u>utoencoder <u>L</u>atent Space Representations. 

TEQUAL is a powerful toolkit designed to streamline the process of training autoencoders as generative models while providing advanced functionalities for analyzing and understanding latent representations. This repository facilitates easy hyperparameter grid searches for optimizing autoencoder performance and introduces innovative features for exploring latent spaces using **topological quotients**.

## Topological Quotients
TEQUAL introduces the concept of topological quotients, allowing you to categorize latent representations based on various topological and geometric features. This provides a novel perspective on understanding the structure of the latent space, aiding in both model interpretation and optimization. The equivalency (or EQ) classes returned by 
these quotients allow you to understand similarities and difference between a wide range of models based on the
structure of the embeddings they produce.

## Features

### 1. Hyperparameter Grid Search
Effortlessly fine-tune your autoencoder model using TEQUAL's built-in hyperparameter grid search functionality. Define a parameter grid in the `params.yaml` file, and TEQUAL will handle the rest, allowing you to identify optimal configurations for training your autoencoders.

### 2. Hyperparameter Compression
Optimize your model by leveraging TEQUAL's hyperparameter compression capabilities. Identify the most influential hyperparameters and streamline your model without compromising performance.

### 3. Hyperparameter Sensitivity Scores
TEQUAL provides a mechanism for computing sensitivity scores for different hyperparameters. Gain insights into how changes in hyperparameter values impact your model's performance, enabling more informed decisions during the optimization process.

### 4. Anomalous Embedding Detection
Detect anomalous embeddings for various models and datasets– Use topological quotients to eliminate or focus your
search to latent representations with "peculiar" topologies and geometries.

## Getting Started

### Installation
Clone the repository and install the required dependencies using Poetry:

```bash
git clone https://github.com/aidos-lab/TEQUAL.git
cd TEQUAL
poetry install
```
Also please include a `.env` that contains the following information:
```
root="path/to/the/installation/location/of/TEQUAL"
params="path/to/the/parameter/driver/params.yaml"
```


### Usage
The pipeline is configured to be run from the `Makefile`. To configure a grid search
based on your `params.yaml` and train the 
desired models on various datasets try running:
```
make setup && make train
```

To compute the topological quotients of your latent representations and allow for
hyperparameter compression, anomaly detection,

## Supported Autoencoders and DataSets

### Configurable Models
- vanilla VAE
- info VAE
- factor VAE
- DIP VAE
- beta VAE
- betatc VAE
- DAE
- WAE MMD


### Datasets
- XYC
- teapots
- MNIST
- FashionMNIST
- CIFAR-10
- CelebA
- LFWPeople
