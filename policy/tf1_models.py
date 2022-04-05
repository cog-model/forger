import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec


class OldDuelingModel(tf.keras.Model):
    def __init__(self, action_dim, reg=1e-6):
        super(OldDuelingModel, self).__init__()
        reg = {'kernel_regularizer': l2(reg), 'bias_regularizer': l2(reg)}

        kernel_init = tf.keras.initializers.VarianceScaling(scale=2.)
        self.h_layer = OldNoisyDense(
            1024, 'relu',
            name='Q_network/dense_1', use_bias=True,
            kernel_initializer=kernel_init,
            **reg
            )

        self.a_head1 = OldNoisyDense(action_dim, name='Q_network/A', use_bias=True, kernel_initializer=kernel_init, **reg)
        self.v_head1 = OldNoisyDense(1, name='Q_network/V', use_bias=True, kernel_initializer=kernel_init, **reg)

    @tf.function
    def call(self, inputs):
        print('Building model')
        features = self.h_layer(inputs)
        advantage, value = tf.split(features, num_or_size_splits=2, axis=-1)
        advantage, value = self.a_head1(advantage), self.v_head1(value)
        advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        out = value + advantage
        return out


class OldConv2D(Conv2D):
    def add_weight(self, name, *args, **kwargs):
        if name == 'kernel':
            name = 'k'
        return super().add_weight(name, *args, **kwargs)


class OldClassicCnn(tf.keras.Model):
    def __init__(self, filters, kernels, strides, activation='relu', reg=1e-6):
        super(OldClassicCnn, self).__init__()
        reg = l2(reg)
        kernel_init = tf.keras.initializers.VarianceScaling(scale=2.)
        self.cnn = Sequential(OldConv2D(
            filters[0],
            kernels[0],
            strides[0],
            activation=activation,
            kernel_regularizer=reg,
            kernel_initializer=kernel_init,
            use_bias=False,
            name='Q_network/conv0/conv/',
            padding='same'
            ))

        for i, (f, k, s) in enumerate(zip(filters[1:], kernels[1:], strides[1:])):
            name = f'Q_network/conv{i + 1}_0/conv/'
            self.cnn.add(OldConv2D(
                f, k, s,
                activation=activation,
                kernel_regularizer=reg,
                kernel_initializer=kernel_init,
                use_bias=False,
                name=name,
                padding='same'
                ))


        self.cnn.add(Flatten())

    @tf.function
    def call(self, inputs):
        return self.cnn(inputs)


class OldNoisyDense(Dense):

    # factorized noise
    def __init__(self, units, *args, **kwargs):
        self.output_dim = units
        self.f = lambda x: tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        super(OldNoisyDense, self).__init__(units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='w_' + self.name[-1],
                                      regularizer=self.kernel_regularizer,
                                      constraint=None)

        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=self.kernel_initializer,
                                            name='w_noise_' + self.name[-1],
                                            regularizer=self.kernel_regularizer,
                                            constraint=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(1, self.units),
                                        initializer=self.bias_initializer,
                                        name='b_' + self.name[-1],
                                        regularizer=self.bias_regularizer,
                                        constraint=None)

            self.bias_sigma = self.add_weight(shape=(1, self.units,),
                                              initializer=self.bias_initializer,
                                              name='b_noise_' + self.name[-1],
                                              regularizer=self.bias_regularizer,
                                              constraint=None)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        if inputs.shape[0]:
            kernel_input = self.f(tf.random.normal(shape=(inputs.shape[0], self.input_dim, 1)))
            kernel_output = self.f(tf.random.normal(shape=(inputs.shape[0], 1, self.units)))
        else:
            kernel_input = self.f(tf.random.normal(shape=(self.input_dim, 1)))
            kernel_output = self.f(tf.random.normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(kernel_input, kernel_output)

        w = self.kernel + self.kernel_sigma * kernel_epsilon

        output = tf.matmul(tf.expand_dims(inputs, axis=1), w)

        if self.use_bias:
            b = self.bias + self.bias_sigma * kernel_output
            output = output + b
        if self.activation is not None:
            output = self.activation(output)
        output = tf.squeeze(output, axis=1)
        return output

