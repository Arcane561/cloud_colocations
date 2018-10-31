"""
The :code:`cloud_net` module provides neural network for the creation
and training of convolutional neural networks (CCNs) for cloud detection and
retrievals.
"""
import numpy as np
import keras
from keras.models               import Sequential
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.core          import Dense, Flatten, Dropout
from keras.layers.pooling       import MaxPooling2D
from keras.layers               import BatchNormalization
from keras.callbacks            import CSVLogger
from keras.optimizers           import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from copy import copy
import os
import pickle

################################################################################
# Custom loss functions
################################################################################

def loss(y_true, y_pred):
    dy = (y_pred[:, 0, :, :] - y_true[:, 0, :, :]) * y_true[:, 1, :, :] * y_true[:, 2, :, :]
    sdy = K.sum(K.square(dy))


    t = y_true[:, 1, :, :] * y_true[:, 2, :, :]
    o = y_pred[:, 1, :, :] * y_true[:, 2, :, :]
    q = K.binary_crossentropy(t,
                              o,
                              from_logits = True)

    return sdy + q

################################################################################
# Custom callbacks.
################################################################################

class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.

    Attributes:

        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.

    """

    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        self.losses += [logs.get('val_loss')]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(
                self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0

            lr = keras.backend.get_value(self.model.optimizer.lr)
            print("\n Reduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])

################################################################################
# CloudNet class hierarchy
################################################################################

class CloudNetBase:

    def load(path):
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dirname = os.path.dirname(path)

        f = open(path, "rb")
        cn = pickle.load(f)
        mp = os.path.join(dirname, cn.model_file)
        cn.model = keras.models.load_model(mp)
        f.close()
        return cn

    def save(self, path):
        f = open(path, "wb")
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dirname = os.path.dirname(path)

        self.model_file = name + "_keras_model"
        model = self.model
        self.model.save(os.path.join(dirname, self.model_file))
        self.model = None
        pickle.dump(self, f)
        f.close()
        self.model = model




class CloudNetSimple(CloudNetBase):
    def __init__(self, dn,
                 layers = 4, kw = 3, filters = 32, filters_max = 128,
                 dense_layers = 2, dense_width = 128,
                 dense_activation = "relu", **kwargs):

        self.dn = dn
        self.channels = [4, 5]

        self.model = Sequential()

        n = 2 * dn + 1
        n_channels = len(self.channels)

        layer_kwargs = {}
        layer_kwargs["filters"] = filters
        layer_kwargs["activation"] = "relu"
        layer_kwargs["kernel_size"] = (kw, kw)
        layer_kwargs["padding"] = "valid"
        layer_kwargs["input_shape"] = (n_channels, n, n)
        layer_kwargs.update(kwargs)


        for i in range(layers):

            self.model.add(Conv2D(**layer_kwargs))
            self.model.add(BatchNormalization(axis = 1))
            np = (kw // 2) * 2
            n -= (kw // 2) * 2
            print(n)

            if (n // 2) > (layers - i) * np:
                self.model.add(MaxPooling2D(pool_size = (2, 2)))
                filters *= 2
                filters = min(filters_max, filters)
                n = n // 2
                layer_kwargs["filters"] = filters

            if i == 0:
                layer_kwargs.pop("input_shape")

        if layers > 0:
            self.model.add(Flatten())
            #self.model.add(Dropout(rate = 0.1))
        else:
            self.model.add(Flatten(input_shape = (n_channels, n, n)))
            #self.model.add(Dropout(rate = 0.1))

        for i in range(dense_layers):
            self.model.add(Dense(dense_width, activation = dense_activation))
            #if i < dense_layers - 1:
                #self.model.add(Dropout(rate = 0.1))

        self.model.add(Dense(1, activation = None))


    def fit(self, x, y):

        logger = CSVLogger("logs/train_log_{0}_{1}.csv".format(self.dn, id(self)), append = True)

        n0 = (x.shape[-1] - 1) // 2
        dn = self.dn

        y = y.reshape(-1, 1)

        x = x[:, self.channels,
              n0 - dn : n0 + dn + 1,
              n0 - dn : n0 + dn + 1]

        self.x_mean = x.mean(axis = (0, 2, 3), keepdims = True)
        self.x_sigma = x.std(axis = (0, 2, 3), keepdims = True)

        x = (x - self.x_mean) / self.x_sigma

        inds = np.random.permutation(x.shape[0])
        n_train = int(0.9 * x.shape[0])
        x_train = x[inds[:n_train], :, :, :]
        y_train = y[inds[:n_train], :]
        x_val   = x[inds[n_train:], :, :, :]
        y_val   = y[inds[n_train:], :]

        print(y.shape)

        #
        # Data generator
        #


        lr_callback = LRDecay(self.model, 2.0, 1e-4, 1)
        self.model.compile(loss = "mean_absolute_error",
                           optimizer = Adam(lr = 0.1)) #SGD(lr = 0.001, momentum = 0.9, decay = 0.00001))

        #self.model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64),
        #                         steps_per_epoch = n_train // 64, epochs = 100,
        #                         validation_data = [x_val, y_val],
        #                         callbacks = [lr_callback, logger])

        self.model.fit(x = x_train, y = y_train, batch_size = 256, epochs = 100,
                       validation_data = [x_val, y_val], callbacks = [lr_callback, logger])

class CloudNet:
    def __init__(self, dn, channels, kw = 3, layers = 2):

        self.channels = channels

        self.model = Sequential()
        n_channels = len(self.channels)

        n = 2 * dn + 1

        layer_kwargs = {"filters" : 32,
                        "kernel_size" : (kw, kw),
                        "activation"  : "relu",
                        "padding" : "same",
                        "input_shape" : (n_channels, n, n)}
        for i in range(layers):
            print(layer_kwargs)
            self.model.add(Conv2D(**layer_kwargs))
            self.model.add(BatchNormalization(axis = 1))
            if i == 0:
                layer_kwargs.pop("input_shape")

        layer_kwargs["filters"] = 2
        layer_kwargs["activation"] = None
        self.model.add(Conv2D(**layer_kwargs))

    def fit(self, x, y, cm, vm):

        x = x[:, self.channels, :, :]

        self.x_mean = x.mean(axis = 0, keepdims = True)
        self.x_sigma = x.std(axis = 0, keepdims = True)

        x = (x - self.x_mean) / self.x_sigma

        y = np.stack([y, cm, vm], axis = 1)
        print(y.shape)

        self.custom_objects = {loss.__name__: loss}
        self.model.compile(loss = loss, optimizer = Adam(lr = 0.1))
        self.model.fit(x = x_train, y = y_train, epochs = 10, batch_size = 512)

    def predict(self, x):
        x = x[:, self.channels, :, :]
        y = self.model.predict((x - self.x_mean) / self.x_sigma)
        return y


################################################################################
# FeatureModel
################################################################################

class FeatureModel:
    """
    The :code:`FeatureModel` class is used to extract the learned features
    from a trained :code:`CloudNet` object.

    After a :code:`FeatureModel` object has been created from a :code:`CloudNet`
    model, it can be applied to any data that has the same number of channels.
    """
    def __init__(self, model):
        """
        Create a :code:`FeatureModel` object from a given :code:`CloudNet` model.

        Arguments:

            model(:code:`CloudNet`): The trained :code:`CloudNet` model from which
                to extract the learned features.
        """
        self.bands = [30, 31]
        self.model = self._make_feature_model(model.model)
        self.n_channels = len(model.channels)
        self.x_mean     = model.x_mean
        self.x_sigma      = model.x_sigma

    def convert_indices(self, inds):
        """
        Converts indices from the input image domain to the spatial domain
        of the filter responses.

        Arguments:

            inds(:code:`numpy.ndarray`): Numpy array containing the indices
            from  the original image domain.

        Returns:

            :code:`numpy.ndarray` of same shape as :code:`inds` with each
            index transformed into the index-coordinates of the filter responses.

        """
        inds = np.copy(inds)

        pixel_width = 1

        for l in model.layers:

            if type(l) == Conv2D:
                inds -= (l.kernel_size // 2 - 1) * pixel_width

            if type(l) == MaxPooling2D:
                inds = inds // 2
                pixel_width *= 2

        return inds


    def _make_feature_model(self,
                            model,
                            input_shape = (2, None, None)):
        """
        Rebuilds only the convolutional layers of a trained CloudNet model
        and copies its weights.

        This can be used to apply the features learned by a CloudNet to
        input data with different shapes.

        Arguments:

            model(CloudNetSimple): The cloud net from which to copy the
                convolutional layers.

            input_shape: The shape of the input to apply the model to.

        """
        fm = Sequential()


        #
        # input layer
        #

        il = model.layers[0]

        fm.add(Conv2D(filters = il.filters,
                    activation = il.activation,
                    kernel_size = il.kernel_size,
                    padding = il.padding,
                    input_shape = input_shape))
        fm.layers[-1].set_weights(il.get_weights())


        for l in model.layers[1:]:

            if type(l) == Flatten:
                break

            if type(l) == Conv2D:
                fm.add(Conv2D(filters = l.filters,
                            activation = l.activation,
                            kernel_size = l.kernel_size,
                            padding = l.padding,
                            input_shape = l.input_shape))

            if type(l) == BatchNormalization:
                fm.add(BatchNormalization(axis = l.axis))

            if type(l) == MaxPooling2D:
                fm.add(MaxPooling2D(pool_size = l.pool_size))

        for l1, l2 in zip(fm.layers, model.layers):
            if not type(l1) == MaxPooling2D:
                l1.set_weights(l2.get_weights())

        return fm

    def apply(self, data):
        """
        Apply filters to input data.

        This function applies the extracted image filters to the given
        input data.

        Arguments:

            data(:code:`numpy.ndarray`): The data to apply the filters to
            given as :code:`numpy.ndarray` with channel along the first axis.

        Returns:

            Filter responses as :code:`numpy.ndarray` with channels along the
            first dimension.

        """
        if not data.shape[0] == self.n_channels:
            Exception("Input data was expected to have shape {0} along its"
                      " first axis, but it actually has {1}."\
                      .format(self.n_channels, data.shape[0]))

        return self.model.predict((data - self.x_mean) / self.x_sigma)



