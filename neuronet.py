# %% importing modules
import numpy as np
import pandas as pd
import tensorflow as tf

# %% defining functions
def weight_variable(shape):
    # create a weight variable with appropriate initialization
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # create a bias variable with appropriate initialization
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class Dense_layer:
    def __init__(self, input_tensor, input_dim, output_dim, layer_name, graph, activation=tf.nn.relu):
        with graph.as_default():
            with tf.name_scope(layer_name):
                with tf.name_scope('weights'):
                    self.weights = weight_variable([input_dim, output_dim])
                    variable_summaries(self.weights)
                with tf.name_scope('biases'):
                    self.biases = bias_variable([output_dim])
                    variable_summaries(self.biases)
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, self.weights) + self.biases
                    tf.summary.histogram('pre_activations', preactivate)
                self.activations = activation(preactivate, name='activation')
                tf.summary.histogram('activations', self.activations)

    @property
    def act(self):
        return self.activations


class Predictor:
    def __init__(self, data_dimensions, target_dimensions):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.dim_x = data_dimensions
        self.dim_y = target_dimensions

        with self.graph.as_default():
            with tf.name_scope('input'):
                self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.dim_x])
                self.y = tf.placeholder(dtype = tf.int64, shape = [None, self.dim_y])

            self.h1 = Dense_layer(self.X, self.dim_x, 10, 'layer1', self.graph)
            self.last = Dense_layer(self.h1.act, 10, self.dim_y, 'layer2', self.graph, activation=tf.nn.softmax)
            self.set_trainer(0.1)

            self.session.run(tf.global_variables_initializer())
            self.combine_summaries()

    def set_trainer(self, learning_rate):
        with self.graph.as_default():
            with tf.name_scope('cross_entropy'):
                with tf.name_scope('total'):
                    self.cross_entropy = tf.losses.softmax_cross_entropy(self.y, self.last.act)
                tf.summary.scalar('cross_entropy', self.cross_entropy)

            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    self.correct_prediction = tf.equal(tf.argmax(self.last.act, 1), self.y)
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

    def combine_summaries(self):
        with self.graph.as_default():
            merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./Tensorboard' + '/train', self.graph)
            self.test_writer = tf.summary.FileWriter('./Tensorboard' + '/test')

    def feed_dict(self, X, y, len):
        with self.graph.as_default():
            return {self.X: np.array(X).reshape(len, self.dim_x), self.y: np.array(y).reshape(len, self.dim_y)}

    def train(self, X, y):
        with self.graph.as_default():
            self.session.run(self.train_step, feed_dict=self.feed_dict(X, y, X.shape[1]))

    def predict(self, X):
        return

# %% testing space
ex = Predictor(6, 3)
X = pd.DataFrame([1,2,3,4,5,6], [7,8,9,10,11,12])
y = pd.DataFrame([0,0,1], [1,0,0])
ex.train(X, y)

# %%
ex.h1.weights
