# Titanic Neural Network Classifier

This repository contains a simple feed-forward neural network implementation designed to classify passengers from the Titanic dataset based on their survival. The goal of this project is to demonstrate a basic neural network model using a structured, supervised learning approach with a well-known dataset. 

## Dataset

The dataset used in this project is the [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset), which contains information on the passengers of the Titanic, including demographics and other relevant information for survival prediction. This dataset is pre-processed and split into training and testing sets using the BuildDataset function in build_dataset.py.

## Project Structure

- **build_dataset.py**: Contains the BuildDataset function, which loads, cleans, encodes, and splits the Titanic dataset for model training and evaluation.
- **titanic_neural_network.py**: Implements a simple neural network class with one hidden layer. The model uses the sigmoid activation function, a basic backpropagation method, and gradient descent for weight updates.
- **example.ipynb**: Demonstrates the entire workflow: dataset loading, model training, and evaluation. This notebook is useful as a quick example for users to get started and test the network on the Titanic dataset.
- **requirements.txt**: Lists the necessary Python packages to run the code.

## Code Description

### build_dataset.py

The BuildDataset function performs the following tasks:
- Loads the Titanic dataset.
- Cleans and fills missing values for age, embarked port, and fare.
- Encodes categorical variables (sex and embarkation port) using one-hot encoding.
- Splits the dataset into training and testing sets.
- Scales the feature values using StandardScaler.

### titanic_neural_network.py

The NeuralNetwork class is a basic feed-forward neural network with:
- One hidden layer.
- Sigmoid activation function.
- Backpropagation for updating weights and biases.
- A simple training loop with an option to monitor loss every 10 epochs.

*Functions in the NeuralNetwork class*:
- forward: Computes the network’s output for a given input.
- backward: Calculates gradients and updates weights/biases.
- train: Iteratively trains the model over a set number of epochs.
- predict: Generates predictions on new data.

### example.ipynb

This notebook includes:
1. Importing and preparing the dataset with BuildDataset.
2. Initializing and training the NeuralNetwork model.
3. Testing and evaluating the model’s accuracy on the test set.

## Usage

### Setup

1. Clone this repository and navigate to the project folder.
2. Install the dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
