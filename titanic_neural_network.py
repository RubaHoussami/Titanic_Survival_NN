import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.hidden_layer_weights = np.random.randn(input_size, hidden_size) # weights from input to hidden layer which are initialized with random numbers from a normal distribution
        self.hidden_layer_bias = np.zeros((1, hidden_size)) # bias for hidden layer which are initialized with zeros
        self.output_layer_weights = np.random.randn(hidden_size, output_size) # weights from hidden to output layer which are initialized with random numbers from a normal distribution
        self.output_layer_bias = np.zeros((1, output_size)) # bias for output layer which are initialized with zeros

    def sigmoid(self, x):
        """
        Sigmoid function used as activation function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Forward pass
        """
        self.z = np.dot(X, self.hidden_layer_weights) + self.hidden_layer_bias # z is the weighted sum of the inputs and the weights (1) plus the bias
        self.a = self.sigmoid(self.z) # a is the activation of the hidden layer which is the sigmoid of z
        self.a_ = np.dot(self.a, self.output_layer_weights) + self.output_layer_bias # a_ is the weighted sum of the hidden layer activations and the weights (2) plus the bias
        self.y_hat = self.sigmoid(self.a_) # y_hat is the activation of the output layer which is the sigmoid of a_
        return self.y_hat
    
    def output_error(self, error):
        """
        Output error
        """
        return np.dot(error, self.output_layer_weights.T) * self.sigmoid_derivative(self.a) # calculate the error of the output layer which is the error multiplied by the derivative of the sigmoid function which is used in backward pass

    def backward(self, X, y):
        """
        Backward pass
        """
        # Calculate the error for the output layer
        error = self.y_hat - y.reshape(-1, 1) # error is the difference between the predicted output and the actual output
        
        # Backpropagate the error
        hidden_output_error = self.output_error(error) # hidden_output_error is the error of the hidden layer which is the output error multiplied by the derivative of the sigmoid function

        # Update weights and biases
        self.output_layer_weights -= self.learning_rate * np.dot(self.a.T, error) # update the weights of the output layer which is the learning rate multiplied by the transpose of the hidden layer activations and the error
        self.output_layer_bias -= self.learning_rate * np.sum(error, axis=0, keepdims=True) # update the bias of the output layer which is the learning rate multiplied by the sum of the error

        self.hidden_layer_weights -= self.learning_rate * np.dot(X.T, hidden_output_error) # update the weights of the hidden layer which is the learning rate multiplied by the transpose of the inputs and the hidden output error
        self.hidden_layer_bias -= self.learning_rate * np.sum(hidden_output_error, axis=0, keepdims=True) # update the bias of the hidden layer which is the learning rate multiplied by the sum of the hidden output error

    def train(self, X, y, epochs=100):
        """
        Train the neural network
        """
        for epoch in epochs:
            # Forward and backward pass
            self.forward(X)
            self.backward(X, y)

            if epoch % 10 == 0:
                loss = -np.mean(y * np.log(self.y_hat + 1e-9) + (1 - y) * np.log(1 - self.y_hat + 1e-9)) # calculate the loss which is the mean of the negative log of the predicted output and the actual output
                print(f'Epoch {epoch}: Loss = {loss}')

    def predict(self, X):
        """
        Predict the output
        """
        output = self.forward(X)
        1 if output >= 0.5 else 0
