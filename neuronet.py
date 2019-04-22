# %% importing modules
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

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


class Predictor:
    def __init__(self, data_dimensions, target_dimensions):
        self.session = tf.Session()
        with tf.name_scope('input'):
            self.X = tf.placeholder(dtype = tf.float32, shape = [None, data_dimensions])
            self.y = tf.placeholder(dtype = tf.int64, shape = [None, target_dimensions])

        self.h1 = nn_layer(self.X, data_dimensions, 10, 'layer1')

        self.last = nn_layer(self.h1, 10, target_dimensions, 'layer2', act=tf.nn.softmax)

        self.set_trainer(0.1)
        self.combine_summaries()

    def set_trainer(self, learning_rate):
        with tf.name_scope('cross_entropy'):
            with tf.name_scope('total'):
                self.cross_entropy = tf.losses.softmax_cross_entropy(self.y, self.last)
            tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.last, 1), self.y)
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

    def combine_summaries(self):
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./Tensorboard' + '/train', self.session.graph)
        test_writer = tf.summary.FileWriter('./Tensorboard' + '/test')
        tf.global_variables_initializer().run(session=self.session)

    def train(self, X, y):
        return

    def predict(self, X):
        return
