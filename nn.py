import numpy as np
import autograd.numpy as anp
from autograd import grad


class LinearClassifier:
    """
    Linear classifier: y = Wx + b
    Network structure: [784, 10]
    """
    def __init__(self, n_input, n_output):
        """
        Initialize linear classifier parameters
        Args:
            n_input: Number of input features (784 for Fashion-MNIST)
            n_output: Number of output classes (10 for Fashion-MNIST)
        """
        self.n_input = n_input
        self.n_output = n_output
        
        # Initialize weights and biases with small random values
        self.W = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros(n_output)
        
    def forward(self, X):
        """
        Forward pass: compute predictions
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Output logits of shape (batch_size, n_output)
        """
        # COMPLETE: Implement linear transformation y = Wx + b
        # 
        # This is a linear classifier that maps input features directly to output logits.
        # You need to:
        # 1. Multiply input X with weight matrix W: X @ W
        #    - X has shape (batch_size, n_input) 
        #    - W has shape (n_input, n_output)
        #    - Result has shape (batch_size, n_output)
        # 2. Add bias term b to each sample
        #    - b has shape (n_output,)
        #    - Broadcasting will handle adding bias to each batch sample
        #
        # Mathematical formula: output = X @ W + b
        # Where @ is matrix multiplication
        #
        # Example:
        # If X is [batch_size=2, n_input=784] and W is [784, 10], b is [10]
        # Then X @ W gives [2, 10] and adding b gives final [2, 10]
        
        #anp.dot (not np.dot) so autograd records the operation for gradient compuation. 
        # b has shape n_output, and will be broadcasted to match the shape of the result of anp.dot(X, self.W)
        return anp.dot(X, self.W) + self.b 
        
        raise NotImplementedError # I Think this can be removed since we have implemented the forward pass.
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + anp.exp(-z))
    
    def softmax(self, z):
        """Softmax activation function"""
        exp_z = anp.exp(z)
        return exp_z / anp.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, X):
        """
        Make predictions using the model
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Predicted class labels
        """
        logits = self.forward(X)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
    
    def get_params(self):
        """Get model parameters as a flat array"""
        return np.concatenate([self.W.flatten(), self.b.flatten()])
    
    def set_params(self, params):
        """Set model parameters from a flat array"""
        W_size = self.n_input * self.n_output
        self.W = params[:W_size].reshape(self.n_input, self.n_output)
        self.b = params[W_size:]


class TwoLayerMLP:
    """
    Two-layer Multi-Layer Perceptron with one hidden layer
    Network structure: [784, 30, 10]
    """
    def __init__(self, n_input, n_hidden, n_output, activation='sigmoid'):
        """
        Initialize MLP parameters
        Args:
            n_input: Number of input features (784)
            n_hidden: Number of hidden units (30)
            n_output: Number of output classes (10)
            activation: Activation function for hidden layer ('sigmoid' or 'relu')
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Initialize weights and biases for both layers
        # First layer: input to hidden
        self.W1 = np.random.randn(n_input, n_hidden) * 0.01
        self.b1 = np.zeros(n_hidden)
        
        # Second layer: hidden to output
        self.W2 = np.random.randn(n_hidden, n_output) * 0.01
        self.b2 = np.zeros(n_output)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + anp.exp(-z))

    def relu(self, z): # ReLU & GeLU Activation are more common in modern environments so we adding them!
        """ReLU activation function"""
        return anp.maximum(0, z)

    def activation(self, z):
        """Activation function for hidden layer (can be sigmoid or ReLU)"""
        if self.activation_function == 'relu':
            return self.relu(z)
        return self.sigmoid(z)
    
    def softmax(self, z):
        """Softmax activation function"""
        exp_z = anp.exp(z)
        return exp_z / anp.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Output logits of shape (batch_size, n_output)
        """
        # COMPLETE: Implement the forward pass for a two-layer neural network
        # 
        # This network has the structure: Input -> Hidden Layer -> Output Layer
        # Network topology: [784] -> [30] -> [10] (for Fashion-MNIST)
        
        # First layer: linear transformation + sigmoid activation
        # COMPLETE: Implement z1 = X @ W1 + b1, then h1 = sigmoid(z1)
        # 
        # Step-by-step:
        # 1. Linear transformation: z1 = X @ W1 + b1
        #    - X has shape (batch_size, n_input) = (batch_size, 784)
        #    - W1 has shape (n_input, n_hidden) = (784, 30)
        #    - b1 has shape (n_hidden,) = (30,)
        #    - z1 will have shape (batch_size, n_hidden) = (batch_size, 30)
        # 2. Apply sigmoid activation: h1 = sigmoid(z1)
        #    - h1 is the output of the hidden layer, same shape as z1
        #
        # Variables to define:
        # z1 = ...  # Linear transformation result
        # h1 = ...  # Activated hidden layer output
        
        # Layer 1: 
        z1 = anp.dot(X, self.W1) + self.b1 # Linear transformation for first layer
        h1 = self.activate(z1) # Apply activation function (sigmoid or ReLU) to get hidden layer output
        
        # Second layer: linear transformation (no activation for output logits)
        # COMPLETE : Implement z2 = h1 @ W2 + b2
        #
        # Step-by-step:
        # 1. Linear transformation: z2 = h1 @ W2 + b2
        #    - h1 has shape (batch_size, n_hidden) = (batch_size, 30)
        #    - W2 has shape (n_hidden, n_output) = (30, 10)
        #    - b2 has shape (n_output,) = (10,)
        #    - z2 will have shape (batch_size, n_output) = (batch_size, 10)
        # 2. z2 represents the final logits (no activation applied here)
        #    - Sigmoid activation will be applied later in predict() or loss function
        #
        # Variable to define:
        # z2 = ...  # Final output logits
        
        # Layer 2: No Activation - Raw Logits and Softmax applied inside the cross-entropy loss function
        # Apply SoftMax here aswell would flatten the gradient and make it harder to compute the loss. So we will apply softmax in the loss function.
        z2 = anp.dot(h1, self.W2) + self.b2 #

        return z2
    
    def predict(self, X):
        """
        Make predictions using the model
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Predicted class labels
        """
        logits = self.forward(X)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
    
    def get_params(self):
        """Get model parameters as a flat array"""
        # Complete : Concatenate all model parameters into a single flat array
        #
        # This function is needed for automatic differentiation (autograd).
        # The grad() function requires all parameters to be in a single array.
        #
        # You need to flatten and concatenate ALL parameters in the correct order:
        # 1. W1: weight matrix of first layer (shape: n_input × n_hidden)
        # 2. b1: bias vector of first layer (shape: n_hidden)
        # 3. W2: weight matrix of second layer (shape: n_hidden × n_output)  
        # 4. b2: bias vector of second layer (shape: n_output)


        #autograd differentites w.r.t. a single array of parameters, so we need to flatten and concatenate all parameters into a single array.
        # The order here must match upacking order W1, b1, W2, b2 in set_params() method.
        return anp.concatenate([self.W1.flatten(), self.b1.flatten(),
                                self.W2.flatten(), self.b2.flatten()])
        
    
    def set_params(self, params):
        """Set model parameters from a flat array"""
        W1_size = self.n_input * self.n_hidden
        b1_size = self.n_hidden
        W2_size = self.n_hidden * self.n_output
        b2_size = self.n_output
        
        idx = 0
        self.W1 = params[idx:idx + W1_size].reshape(self.n_input, self.n_hidden)
        idx += W1_size
        
        self.b1 = params[idx:idx + b1_size]
        idx += b1_size
        
        self.W2 = params[idx:idx + W2_size].reshape(self.n_hidden, self.n_output)
        idx += W2_size
        
        self.b2 = params[idx:idx + b2_size]


class ThreeLayerMLP:
    """
    Three-layer Multi-Layer Perceptron with two hidden layers
    Network structure: [784, 30, 30, 10] (default)
    """
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, use_residual=False, activation = 'sigmoid'):
        """
        Initialize MLP parameters
        Args:
            n_input: Number of input features (784)
            n_hidden1: Number of first hidden layer units (30)
            n_hidden2: Number of second hidden layer units (30)
            n_output: Number of output classes (10)
            use_residual: Whether to use residual connections
            activation: Activation function for hidden layers ('sigmoid' or 'relu')
        """
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.use_residual = use_residual
        self.activation_function = activation #activation goes last so the existing postional calls keep working
        
        # Check if residual connection is possible (hidden layers must have same size)
        if use_residual and n_hidden1 != n_hidden2:
            raise ValueError("For residual connections, hidden layers must have the same size")
        
        # Initialize weights and biases for all three layers
        # First layer: input to hidden1
        self.W1 = np.random.randn(n_input, n_hidden1) * 0.01
        self.b1 = np.zeros(n_hidden1)
        
        # Second layer: hidden1 to hidden2
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * 0.01
        self.b2 = np.zeros(n_hidden2)
        
        # Third layer: hidden2 to output
        self.W3 = np.random.randn(n_hidden2, n_output) * 0.01
        self.b3 = np.zeros(n_output)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + anp.exp(-z))

    #Added Helper functions for ReLU and Activation selection
    def relu(self, z):
        """ReLU activation function"""
        return anp.maximum(0, z)

    def activate(self, z):
        """Activation function for hidden layers (can be sigmoid or ReLU)"""
        if self.activation_function == 'relu':
            return self.relu(z)
        return self.sigmoid(z)
        
    def softmax(self, z):
        """Softmax activation function"""
        exp_z = anp.exp(z)
        return exp_z / anp.sum(exp_z, axis=1, keepdims=True)


    # Need to add some comments to explain the forward pass and residual connections in my own words here not just 
    def forward(self, X):
        """
        Forward pass through the network
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Output logits of shape (batch_size, n_output)
        """
        # COMPLETE : Implement the forward pass for a three-layer neural network
        # 
        # This network has the structure: Input -> Hidden1 -> Hidden2 -> Output
        # Network topology: [784] -> [30] -> [30] -> [10] (for Fashion-MNIST)
        # With optional residual connections between hidden layers
        
        # First layer: linear transformation + sigmoid activation
        # COMPLETE: Implement z1 = X @ W1 + b1, then h1 = sigmoid(z1)
        # 
        # Step-by-step:
        # 1. Linear transformation: z1 = X @ W1 + b1
        #    - X has shape (batch_size, n_input) = (batch_size, 784)
        #    - W1 has shape (n_input, n_hidden1) = (784, 30)
        #    - b1 has shape (n_hidden1,) = (30,)
        #    - z1 will have shape (batch_size, n_hidden1) = (batch_size, 30)
        # 2. Apply sigmoid activation: h1 = sigmoid(z1)
        
        z1 = anp.dot(X, self.W1) + self.b1 # Linear transformation for first layer
        h1 = self.activate(z1) # Apply activation function (sigmoid or ReLU) to get hidden layer output

        
        # Second layer: linear transformation + sigmoid activation
        # COMPLETE: Implement z2 = h1 @ W2 + b2, then h2_raw = sigmoid(z2)
        #
        # Step-by-step:
        # 1. Linear transformation: z2 = h1 @ W2 + b2
        #    - h1 has shape (batch_size, n_hidden1) = (batch_size, 30)
        #    - W2 has shape (n_hidden1, n_hidden2) = (30, 30)
        #    - b2 has shape (n_hidden2,) = (30,)
        #    - z2 will have shape (batch_size, n_hidden2) = (batch_size, 30)
        # 2. Apply sigmoid activation: h2_raw = sigmoid(z2)
        
        z2 = anp.dot(h1, self.W2) + self.b2 # Linear transformation for second layer
        h2_raw = self.activate(z2) # Apply activation function (sigmoid or ReLu)

        # Apply residual connection if enabled
        # COMPLETE: Implement residual connection logic
        #
        # Residual connections help with gradient flow in deep networks.
        # The idea is to add the input of a layer to its output: h2 = h2_raw + h1
        # This requires h1 and h2_raw to have the same shape (n_hidden1 == n_hidden2)
        #
        # If residual connections are enabled (self.use_residual == True):
        #   h2 = h2_raw + h1  # Add skip connection from previous layer
        # Else:
        #   h2 = h2_raw       # Use normal activation without skip connection
        
        if self.use_residual:
            h2 = h2_raw + h1  # Residual connection: add previous layer output
        else:
            h2 = h2_raw       # No residual connection: use normal activation

        # Third layer: linear transformation (output layer)
        # COMPLETE: Implement z3 = h2 @ W3 + b3
        
        z3 = anp.dot(h2, self.W3) + self.b3
        return z3
    
    def predict(self, X):
        """
        Make predictions using the model
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Predicted class labels
        """
        logits = self.forward(X)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
    
    def get_params(self):
        """Get model parameters as a flat array"""
        # COMPLETE: Concatenate all model parameters into a single flat array
        #
        # This function is needed for automatic differentiation (autograd).
        # The grad() function requires all parameters to be in a single array.
        
        # Flatten all six parameters (W1, b1, W2, b2, W3, b3) and concatenate them into a single array
        return anp.concatenate([self.W1.flatten(), self.b1.flatten(),
                                self.W2.flatten(), self.b2.flatten(),
                                self.W3.flatten(), self.b3.flatten()])
    
    def set_params(self, params):
        """Set model parameters from a flat array"""
        W1_size = self.n_input * self.n_hidden1
        b1_size = self.n_hidden1
        W2_size = self.n_hidden1 * self.n_hidden2
        b2_size = self.n_hidden2
        W3_size = self.n_hidden2 * self.n_output
        b3_size = self.n_output
        
        idx = 0
        self.W1 = params[idx:idx + W1_size].reshape(self.n_input, self.n_hidden1)
        idx += W1_size
        
        self.b1 = params[idx:idx + b1_size]
        idx += b1_size
        
        self.W2 = params[idx:idx + W2_size].reshape(self.n_hidden1, self.n_hidden2)
        idx += W2_size
        
        self.b2 = params[idx:idx + b2_size]
        idx += b2_size
        
        self.W3 = params[idx:idx + W3_size].reshape(self.n_hidden2, self.n_output)
        idx += W3_size
        
        self.b3 = params[idx:idx + b3_size]
