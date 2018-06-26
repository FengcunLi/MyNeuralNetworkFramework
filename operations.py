import numpy as np

class xavier_initializer(object):
    """docstring for xavier_initializer"""
    def __init__(self):
        super(xavier_initializer, self).__init__()

    def __call__(self, shapes):
        return np.random.randn(*shapes) / np.sqrt(shapes[0])

class zero_initializer(object):
    """docstring for zero_initializer"""
    def __init__(self):
        super(zero_initializer, self).__init__()

    def __call__(self, shapes):
        return np.zeros(shapes)

class FullyConnected(object):
    def __init__(self, num_inputs, num_outputs, 
        weights_initializer=xavier_initializer(), biases_initializer=zero_initializer()):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.w = self.weights_initializer((num_inputs, num_outputs))
        self.b = self.biases_initializer((1, num_outputs))

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.w) + self.b
    
    def backward(self, upstream_gradients):
        self.x_local_gradients = self.w.T
        self.w_local_gradients = self.x.T
        self.b_local_gradients = np.ones((self.x.shape[0], 1)).T

        self.x_gradients = np.dot(upstream_gradients, self.x_local_gradients)
        self.w_gradients = np.dot(self.w_local_gradients, upstream_gradients)
        self.b_gradients = np.dot(self.b_local_gradients, upstream_gradients)
        return self.x_gradients

    def apply_gradients(self, learning_rate=1e-4):
        self.w -= self.w_gradients * learning_rate
        self.b -= self.b_gradients * learning_rate

class ReLU(object):
    """docstring for ReLU"""
    def __init__(self):
        super(ReLU, self).__init__()
    
    def forward(self, x):
        self.x = x
        return np.clip(self.x, 0, a_max=None)

    def backward(self, upstream_gradients):
        self.local_gradients = (self.x > 0).astype(float)
        return self.local_gradients * upstream_gradients

class Sigmoid(object):
    """docstring for Sigmoid"""
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.x = x
        self.sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid_x

    def backward(self, upstream_gradients):
        self.local_gradients = self.sigmoid_x * (1. - self.sigmoid_x)
        return self.local_gradients * upstream_gradients

class SquaredError(object):
    """docstring for SquaredError"""
    def __init__(self):
        super(SquaredError, self).__init__()

    def forward(self, target, prediction):
        self.target = target
        self.prediction = prediction
        return (target - prediction)**2

    def backward(self, upstream_gradients):
        self.target_local_gradients = 2 * (self.target - self.prediction)
        self.prediction_local_gradients = -2 * (self.target - self.prediction)
        self.target_gradients = self.target_local_gradients * upstream_gradients
        self.prediction_gradients = self.prediction_local_gradients * upstream_gradients
        return self.prediction_gradients

class Sum(object):
    """docstring for Sum"""
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x):
        self.x = x
        return np.sum(self.x)

    def backward(self, upstream_gradients):
        self.local_gradients = np.ones_like(self.x)
        return self.local_gradients * upstream_gradients

class Multiply(object):
    """docstring for Multiply"""
    def __init__(self):
        super(Multiply, self).__init__()
    
    def forward(self, x_1, x_2):
        self.x_1 = x_1
        self.x_2 = x_2
        return self.x_1 * self.x_2

    def backward(self, upstream_gradients):
        self.x_1_local_gradients = self.x_2
        self.x_2_local_gradients = self.x_1
        self.x_1_gradients = upstream_gradients * self.x_1_local_gradients
        self.x_2_gradients = upstream_gradients * self.x_2_local_gradients
        return self.x_1_gradients, self.x_2_gradients

class SoftMax(object):
    """docstring for SoftMax"""
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, x):
        self.x = x
        return np.exp(self.x) / np.sum(np.exp(self.x), axis=1)

    def backward(self, upstream_gradients):
        # TO DO
        pass