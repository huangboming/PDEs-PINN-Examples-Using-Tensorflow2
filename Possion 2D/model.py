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
      train_min, train_max: the range of the PDE
      weights: a list that contains weights of each layer
      
    You can call the model to calculate the result based on given dataset
    """
    def __init__(self, layers, train_min, train_max):
        self.layers = layers
        self.train_min = train_min
        self.train_max = train_max
        
    def __call__(self, training_set):
        x = training_set
        x = 2.0 * (x - self.train_min) / (self.train_max - self.train_min) - 1.0
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
    
def train_step(model, optimizer, training_set, b_x_ymin, b_x_ymax, b_xmin_y, b_xmax_y, w1=1, w2=1):
    """
    One training step.
    
    Paras:
      model: Neural network model
      optimizer: tensorflow optimizer
      training_set: training set of pde
      b_x_ymin, b_x_ymax, b_xmin_y, b_xmax_y: training set of boundary
      w1, w2: weights of loss_pde and loss_boundary
      
    Return:
      loss score at current step
    """
    
    # calculate loss
    with tf.GradientTape() as utter_tape:
        with tf.GradientTape(persistent=True) as outer_tape:
            with tf.GradientTape(persistent=True) as inner_tape:
                x = training_set[:, 0]
                y = training_set[:, 1]
                u = model(tf.stack([x, y], axis=1))
            u_x = inner_tape.gradient(u, x)
            u_y = inner_tape.gradient(u, y)
        u_xx = outer_tape.gradient(u_x, x)
        u_yy = outer_tape.gradient(u_y, y)
        
        f = 2 * np.pi**2 * tf.sin(np.pi * training_set[:, 0]) * tf.sin(np.pi * training_set[:, 1])
        loss_f_array = -(u_xx + u_yy) - f
        loss_f = tf.reduce_mean(tf.square(loss_f_array))
    
        b_x_ymin_pred = model(b_x_ymin)
        b_x_ymax_pred = model(b_x_ymax)
        b_xmin_y_pred = model(b_xmin_y)
        b_xmax_y_pred = model(b_xmax_y)
    
        loss_b = (tf.reduce_mean(tf.square(b_x_ymin_pred)) + 
                  tf.reduce_mean(tf.square(b_x_ymax_pred)) + 
                  tf.reduce_mean(tf.square(b_xmin_y_pred)) + 
                  tf.reduce_mean(tf.square(b_xmax_y_pred)))
    
        loss = w1 * loss_f + w2 * loss_b
        
    # optimize with optimizer
    optimizer.minimize(loss, model.weights, tape=utter_tape)
    return loss

def train_model(model, training_set, b_x_ymin, b_x_ymax, b_xmin_y, b_xmax_y, w1=1, w2=1, epochs=1000, learning_rate=0.0008, warning=False):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for epoch in range(epochs):
        loss = train_step(model, optimizer, training_set, b_x_ymin, b_x_ymax, b_xmin_y, b_xmax_y, w1, w2)
        history_loss.append(loss)
        if warning:
            print(f'epoch: {epoch + 1}, loss: {loss:.5f}')  
            
    return history_loss