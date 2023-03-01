import tensorflow as tf
import numpy as np

class Layer:
    """
    Define the structure of every layer.
    
    Paras:
      input_dim: size of input dataset
      output_dim: number of neurons
      activation: activation function, default: None(i.e. without activation function)
      
    Properties:
      activation: activation function
      W, b: weights of the layer
      weights: a list that contains W and b
      
    You can just call a layer to execute the progress called "forward pass"
    """
    def __init__(self, input_dim, output_dim, activation=None):
        self.activation = activation
        
        # initial weights
        std = np.sqrt(2 / (input_dim + output_dim))
        self.W = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, output_dim), stddev=std))
        self.b = tf.Variable(initial_value=tf.zeros(shape=(output_dim, )))
        
    def __call__(self, inputs):
        # forward pass
        if self.activation:
            return self.activation(tf.matmul(inputs, self.W) + self.b)
        else:
            # If the layer don't have an activation, go linear combination directly
            return tf.matmul(inputs, self.W) + self.b
        
    @property
    def weights(self):
        # define "weight" property
        return [self.W, self.b]
        
class Model:
    """
    Define the model
    
    Paras:
      layers: a list of Layer
      xmin, xmax: the range of the PDE
      
    Properties:
      layers: a list of Layer
      xmin, xmax: the range of the PDE
      weights: a list that contains weights of each layer
      
    You can call the model to calculate the result based on given dataset
    """
    def __init__(self, layers, xmin, xmax):
        self.layers = layers
        self.xmin = xmin
        self.xmax = xmax
        
    def __call__(self, training_set):
        x = training_set
        x = 2.0 * (x - self.xmin) / (self.xmax - self.xmin) - 1.0
        for layer in self.layers:
            # update inputs
            x = layer(x)
        return x
    
    @property
    def weights(self):
        # 将所有layer的参数整合在一起
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
    
    
# functions to train the model

def calculate_gradient_and_loss(model, x, b0, b1, w1, w2):
    """
    Calculate loss score of the model and gradients of each weight.
    
    Paras: 
      model: a neural network model
      x: dataset of the PDE
      b0, b1: boundary condiction dataset
      w1, w2: weights of loss function of the equation and loss function of BC
      
    Return value:
      gradients: a list that contains gradient wrt every parameter in the model
      loss: loss score at the step
    """
    
    with tf.GradientTape() as utter_tape:
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                u = model(x)
            u_x = inner_tape.gradient(u, x)
        u_xx = outer_tape.gradient(u_x, x)
    
        # loss function of the equation
        f = np.pi**2 * tf.sin(np.pi * x)
        loss_f_array = -u_xx - f
        loss_f = tf.reduce_mean(tf.square(loss_f_array))
    
        # loss function of BC
        b0_pred = model(b0)
        b1_pred = model(b1)
        loss_b = tf.reduce_mean(tf.square(b0_pred)) + tf.reduce_mean(tf.square(b1_pred))
    
        # total loss function
        loss = w1 * loss_f + w2 * loss_b
        
    # calculate gradients
    gradients = utter_tape.gradient(loss, model.weights)
    return gradients, loss

def update_weights(gradients, weights, learning_rate=0.1):
    """
    Update model weights by stochastic gradient descent(SGD)
    
    Paras:
      gradients: list of gradients calculated by tf.GradientTape().gradient
      weights: list of weights
      learning_rate: learning rate
    """
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)
        
def train_model(model, x, b0, b1, w1, w2, epochs=1000, learning_rate=0.0008, warning=False):
    """
    Train neural network model.
    
    Paras: 
      model: a neural network model
      x: dataset of the PDE
      b0, b1: boundary condiction dataset
      w1, w2: weights of loss function of the equation and loss function of BC
      epochs: epochs, i.e. times of training, default: 1000
      learning_rate: learning rate of SGD, default: 0.0008
      warning: whether to show loss score of each epochs, default: False
      
    Return value:
      history_loss: a list that contains loss score at each step
      
    """
    
    history_loss = []
    for epoch in range(epochs):
        gradients, loss = calculate_gradient_and_loss(model, x, b0, b1, w1, w2)
        update_weights(gradients, model.weights, learning_rate=learning_rate)
        
        history_loss.append(loss)
        if warning:
            print(f'epoch: {epoch + 1}, loss: {loss:.5f}')
            
            
    return history_loss
        
