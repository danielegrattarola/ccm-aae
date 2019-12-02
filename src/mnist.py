import sys
import time
import warnings

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Input, Lambda, LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from spektral.utils import batch_iterator
from spektral.utils.logging import init_logging

from geometry import ccm_normal, ccm_uniform, clip, get_distance, CCMMembership

# Keras 2.2.2 throws UserWarnings all over the place during training
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1, image_size * image_size)
x_test = x_test.reshape(-1, image_size * image_size)

# Parameters
input_shape = (image_size ** 2, )  # Shape of the network's input
batch_size = 32            # Number of samples in a minibatch
epochs = 1000              # Number of training epochs
es_patience = 50           # Early stopping patience
radius = 1                 # Radius of the CCM
disc_weights = [0.5, 0.5]  # Weights for averaging discriminators
latent_dim = 21            # Dimension of the latent/ambient space
scale = 1                  # Variance of the prior
sigma = 5                  # Width of the membership
logdir = init_logging()    # Create log directory and file

if radius == 0:
    latent_dim -= 1  # Euclidean manifold uses intrinsic coordinates

# Train/Val split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5000)

# Encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Lambda(lambda _: K.cast(K.greater(_, K.random_uniform(K.shape(_))), _.dtype))(inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
z = Dense(latent_dim, name='z')(x)
encoder = Model(inputs, z, name='encoder')

# Decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(128, activation='relu')(latent_inputs)
x = Dense(256, activation='relu')(x)
outputs = Dense(input_shape[0], activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

# Autoencoder
outputs = decoder(encoder(inputs))
model = Model(inputs, outputs, name='ae')
model.compile(optimizer='adam', loss='binary_crossentropy')

# Discriminator
d_in = Input(shape=(latent_dim,))
d_1 = Dense(64, kernel_regularizer=l2())(d_in)
d_1 = LeakyReLU()(d_1)
d_1 = Dense(64, kernel_regularizer=l2())(d_1)
d_1 = LeakyReLU()(d_1)
d_1 = Dense(1, activation='sigmoid', kernel_regularizer=l2())(d_1)
discriminator = Model(d_in, d_1, name='discriminator')
discriminator.compile('adam', 'binary_crossentropy', [mean_pred])

# Frozen discriminator + encoder + membership function
discriminator.trainable = False
d_1 = discriminator(encoder(inputs))
d_2 = CCMMembership(r=radius, sigma=sigma)(encoder(inputs))
d_out = Lambda(lambda x_: disc_weights[0] * x_[0] + disc_weights[1] * x_[1])([d_1, d_2])
enc_discriminator = Model(inputs, d_out, name='enc_discriminator')
enc_discriminator.compile('adam', 'binary_crossentropy', [mean_pred])

# Fit model
print('Fitting CCM-AAE')
t = time.time()
current_batch = 0
model_loss = 0
adv_loss_neg = 0
adv_loss_pos = 0
adv_acc_neg = 0
adv_acc_pos = 0
adv_fool = 0
best_val_loss = np.inf
patience = es_patience
batches_in_epoch = 1 + x_train.shape[0] // batch_size
total_batches = batches_in_epoch * epochs

zeros = np.zeros(x_train.shape[0])
ones = np.ones(x_train.shape[0])
for batch, z, o in batch_iterator([x_train, zeros, ones], batch_size=batch_size, epochs=epochs):
    model_loss += model.train_on_batch(batch, batch)

    # Regularization
    batch_x = encoder.predict(batch)
    adv_res_neg = discriminator.train_on_batch(batch_x, z)

    if radius > 0:
        batch_x = ccm_uniform(batch.shape[0], dim=latent_dim, r=radius)
    else:
        batch_x = ccm_normal(batch.shape[0], dim=latent_dim, r=radius, scale=scale)
    adv_res_pos = discriminator.train_on_batch(batch_x, o)

    # Fit encoder to fool discriminator
    adv_res_fool = enc_discriminator.train_on_batch(batch, o)

    # Update stats
    adv_loss_neg += adv_res_neg[0]
    adv_loss_pos += adv_res_pos[0]
    adv_acc_neg += adv_res_neg[1]
    adv_acc_pos += adv_res_pos[1]
    adv_fool += adv_res_fool[1]
    current_batch += 1
    if current_batch % batches_in_epoch == 0:
        model_loss /= batches_in_epoch
        adv_loss_neg /= batches_in_epoch
        adv_loss_pos /= batches_in_epoch
        adv_acc_neg /= batches_in_epoch
        adv_acc_pos /= batches_in_epoch
        adv_fool /= batches_in_epoch
        model_val_loss = model.evaluate(x_val, x_val, batch_size=batch_size, verbose=0)
        print('Epoch {:3d} ({:2.2f}s) - '
              'loss {:.2f} - val_loss {:.2f} - '
              'adv_loss {:.2f} & {:.2f} - '
              'adv_acc {:.2f} ({:.2f} & {:.2f}) - '
              'adv_fool (should be {:.2f}): {:.2f}'
              ''.format(current_batch // batches_in_epoch, time.time() - t,
                        model_loss, model_val_loss,
                        adv_loss_neg, adv_loss_pos,
                        (adv_acc_neg + adv_acc_pos) / 2,
                        adv_acc_neg, adv_acc_pos,
                        disc_weights[0] * 0.5 + disc_weights[1],
                        adv_fool))
        if model_val_loss < best_val_loss:
            best_val_loss = model_val_loss
            patience = es_patience
            print('New best val_loss {:.3f}'.format(best_val_loss))
            model.save_weights(logdir + 'model_best_val_weights.h5')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping (best val_loss: {})'.format(best_val_loss))
                break

        t = time.time()
        model_loss = 0
        adv_loss_neg = 0
        adv_loss_pos = 0
        adv_acc_neg = 0
        adv_acc_pos = 0
        adv_fool = 0

# Post-training
print('Loading best weights')
model.load_weights(logdir + 'model_best_val_weights.h5')
test_loss = model.evaluate(x_test, x_test, batch_size=batch_size, verbose=0)
print('Test loss: {:.2f}'.format(test_loss))


# KNN test
def sample_points(data_, labels_, n_):
    target_idxs_ = []
    for l_ in np.unique(labels_):
        t_ = np.argwhere(labels_ == l_).reshape(-1)
        np.random.shuffle(t_)
        target_idxs_.append(t_[:n_])
    target_idxs_ = np.hstack(target_idxs_)
    return data_[target_idxs_], labels_[target_idxs_]


embeddings_train = clip(encoder.predict(x_train), radius)
embeddings_test = clip(encoder.predict(x_test), radius)

knn = KNeighborsClassifier(n_neighbors=5, metric=get_distance(radius))

for n_labels in [100, 600, 1000]:
    knn.fit(*sample_points(embeddings_train, y_train, n_labels))
    knn_score = knn.score(embeddings_test, y_test)
    print('KNN score ({} labels): {}'.format(n_labels, knn_score))
