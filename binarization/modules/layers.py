import numpy as np

# dense layer default
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.backend import variable

from keras.engine.topology import Layer
# from keras.layers import Dense

class LRN(Layer):
    def __init__(self, **kwargs):
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.input_shapes = input_shapes
        self.pad = K.zeros(shape=(self.input_shapes[0], 1, 1, self.input_shapes[2], self.input_shapes[3]))

    def call(self, x):
        #
        # x: N x C x H x W
        # pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        # pad = K.zeros(shape=(x.get_shape().as_list()[0], 1, 1, x.get_shape().as_list()[2], x.get_shape().as_list()[3]))
        # pad = K.zeros(shape=(self.input_shapes[0], 1, 1, self.input_shapes[2], self.input_shapes[3]))
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = K.concatenate((K.concatenate((x_sq,pad,pad,pad,pad),axis=2),
                            K.concatenate((pad,x_sq,pad,pad,pad),axis=2),
                            K.concatenate((pad,pad,x_sq,pad,pad),axis=2),
                            K.concatenate((pad,pad,pad,x_sq,pad),axis=2),
                            K.concatenate((pad,pad,pad,pad,x_sq),axis=2)),axis=1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x

class Branches(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    # @interfaces.legacy_dense_support
    def __init__(self, num_branches, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Branches, self).__init__(**kwargs)
        
        self.num_branches = num_branches

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = [None]*self.num_branches
        self.bias = [None]*self.num_branches

        # set kernel
        for i in range(self.num_branches):
            self.kernel[i] = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel_'+str(i),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=False)
        
            if self.use_bias:
                self.bias[i] = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias_'+str(i),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=False)
            
            else:
                self.bias[i] = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        # find current epoch
        file = open('k.txt', 'r')
        k_str = file.read()
        file.close()
        k = int(k_str)

        # set all trainable to false
        for i in range(self.num_branches):
            self.kernel[i].trainalbe = False
            self.bias[i].trainable = False

        # set k trainable to true
        self.kernel[k].trainable = True
        self.bias[k].trainable = True
        
        # computation
        output = K.dot(inputs, self.kernel[k])
        if self.use_bias:
            output = K.bias_add(output, self.bias[k])
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Branches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))