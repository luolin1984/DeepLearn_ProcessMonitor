import keras
from keras import backend as K
from keras.engine.topology import Layer

class Constraint(object):
    """
    Constraint template
    """

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MinMax(Constraint):
    """
    Customized min-max constraint for scalar
    """

    def __init__(self, min_value=0.0, max_value=10.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

def nmodeproduct(x, w, mode):
    """
    n-mode product for 2D matrices
    x: NxHxW
    mode=1 -> w: Hxh
    mode=2 -> w: Wxw

    output: NxhxW (mode1) or NxHxw (mode2)
    """
    if mode == 2:
        x = K.dot(x, w)
    else:
        x = K.permute_dimensions(x, (0, 2, 1))
        x = K.dot(x, w)
        x = K.permute_dimensions(x, (0, 2, 1))
    return x

class orthLayer(Layer):
    """
    Orthogonal Layer
    """

    def __init__(self, output_dim,
                 kernel_regularizer=None,
                 kernel_constraint=None, **kwargs):

        self.output_dim = output_dim
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        super(orthLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', shape=(input_shape[1], self.output_dim[0]),
                                  initializer='glorot_normal',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(input_shape[2], self.output_dim[1]),
                                  initializer='glorot_normal',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True)

        self.bias = self.add_weight(name='bias', shape=(self.output_dim[0], self.output_dim[1]),
                                    initializer='zeros', trainable=True)
        super(orthLayer, self).build(input_shape)

    def call(self, x):
        print(K.int_shape(x))
        x = nmodeproduct(x, self.W1, 1)
        x = nmodeproduct(x, self.W2, 2)
        x = K.bias_add(x, self.bias)

        if self.output_dim[1] == 1:
            x = K.squeeze(x, axis=-1)
        print(K.int_shape(x))
        return x

    def compute_output_shape(self, input_shape):
        if self.output_dim[1] == 1:
            return (input_shape[0], self.output_dim[0])
        else:
            return (input_shape[0], self.output_dim[0], self.output_dim[1])



def OC(template, dropout=0.1, regularizer=None, constraint=None):

    print(template)
    inputs = keras.layers.Input(template[0])

    x = inputs
    for k in range(1, len(template) - 1):
        x = orthLayer(template[k], regularizer, constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    x = orthLayer(template[-1], regularizer, constraint)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(0.001)

    model.compile(optimizer, 'categorical_crossentropy', ['acc', ])

    return model