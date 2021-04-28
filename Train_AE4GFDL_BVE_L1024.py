# Convolutional Variational Autoencoder for Field Reconstruction

# This script read simulation data and pass them through forward operators to produce 'observation'
# Then observation and truth can be paired to train CVAE to reconstruct the full field from incomplete
# observation.


# Import libraries

import os
import time
import tensorflow as tf
import netCDF4 as nc4
import numpy as np


# Load the model dataset, scale, and mask

dsFile = nc4.Dataset("train_barotropic.nc")
stream0 = dsFile.variables["stream"]
streamObs = np.copy(stream0[400:, :, :])
streamStd = np.std(streamObs)
streamAve = np.mean(streamObs)
print("||| streamStd: {:.6f}".format(streamStd))
print("||| streamAve: {:.6f}".format(streamAve))
# ||| streamStd: 6944601.500000
# ||| streamAve: -1021574.250000
# normalize streamfunction
psi0 = (streamObs - streamAve) / streamStd
print("||| streamMax: {}".format(np.max(psi0)))
print("||| streamMin: {}".format(np.min(psi0)))
vort0 = dsFile.variables["vor"]
vort = np.copy(vort0[400:, :, :])
vortSH = -np.copy(vort0[400:, 0:49, :])
vort[:, 0:49, :] = np.copy(vortSH)
psi0masked = np.where(vort > 5e-6, -100.0, psi0)
psi0masked[:, 0:49, :] = np.where(
    vort[:, 0:49, :] > 5e-6, 100.0, psi0[:, 0:49, :])
fraction = np.sum(np.abs(psi0masked) > 99.0) / psi0masked.size
print("||| fractionALL: {:.2f}%".format(fraction * 100))
fractionNH = np.sum(
    np.abs(psi0masked[:, 48:, :]) > 99.0) / (psi0masked.size / 2)
print("||| fraction_NH: {:.2f}%".format(fractionNH * 100))
# ||| fractionALL: 15.60%
# ||| fraction_NH: 30.19%
lat = dsFile.variables["lat"]
xLat = np.cos(np.deg2rad(lat)).reshape(1, psi0.shape[1], 1, 1)

# shuffle and partition data
psi0 = psi0.reshape(psi0.shape[0], psi0.shape[1], psi0.shape[2], 1)
psi0masked = psi0masked.reshape(psi0.shape[0], psi0.shape[1], psi0.shape[2], 1)
idx = np.arange(psi0.shape[0])
np.random.shuffle(idx)

trainObs_images = np.copy(psi0masked[idx[0:65520], :, :, :])
trainFul_images = np.copy(psi0[idx[0:65520], :, :, :])
testObs_images = np.copy(psi0masked[idx[65520:], :, :, :])
testFul_images = np.copy(psi0[idx[65520:], :, :, :])

del stream0, streamObs, psi0, psi0masked, idx, dsFile, vort, vortSH

# Set up DL dataset

train_size = 65520
batch_size = 64
test_size = 7280  # 10% as test set

trainObs_dataset = tf.data.Dataset.from_tensor_slices(
    trainObs_images).batch(batch_size)
trainFul_dataset = tf.data.Dataset.from_tensor_slices(
    trainFul_images).batch(batch_size)
testObs_dataset = tf.data.Dataset.from_tensor_slices(
    testObs_images).batch(batch_size)
testFul_dataset = tf.data.Dataset.from_tensor_slices(
    testFul_images).batch(batch_size)

# Define the encoder and decoder networks


class AutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for
    training.
    """
    def __init__(self, latent_dim=1024, name="autoencoder", **kwargs):
        """
        Args:
            latent_dim:
            name:
            **kwargs:
        """
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(96, 192, 1)),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=1024,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=2048,
                    kernel_size=3,
                    strides=(1, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=1024,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim),
                tf.keras.layers.Reshape((1, 1, latent_dim)),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(1, 1, latent_dim)),
                tf.keras.layers.Conv2DTranspose(
                    filters=1024,
                    kernel_size=3,
                    strides=(3, 3),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1024,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=2048,
                    kernel_size=3,
                    strides=(1, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=512,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1024,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=256,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=512,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=256,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=1,
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="same"
                ),
            ]
        )

    def call(self, x, **kwargs):
        """
        Args:
            x:
            **kwargs:
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the loss function and the optimizer
def compute_loss(model, xObs, xFul):
    """
    Args:
        model:
        xObs:
        xFul:
    """
    xRec = model(xObs)
    ssimLoss = 1.0 - tf.image.ssim_multiscale(xFul, xRec, 25.0, filter_size=6)
    squares = (xRec - xFul) ** 2
    reconstrLoss = tf.reduce_mean(squares, axis=[1, 2, 3])
    totalLoss = 0.5 * reconstrLoss + 0.5 * ssimLoss
    return tf.reduce_mean(totalLoss)


def compute_loss2(model, xObs, xFul):
    xRec = model(xObs)
    ssimLoss = 1.0 - tf.image.ssim_multiscale(xFul, xRec, 25.0, filter_size=6)
    squares = (xRec - xFul) ** 2
    reconstrLoss = tf.reduce_mean(squares, axis=[1, 2, 3])
    totalLoss = 0.5 * reconstrLoss + 0.5 * ssimLoss
    return (
        tf.reduce_mean(totalLoss),
        tf.reduce_mean(0.5 * reconstrLoss),
        tf.reduce_mean(0.5 * ssimLoss),
    )


@tf.function
def train_step(model, xObs, xFul, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.

    Args:
        model:
        xObs:
        xFul:
        optimizer:
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, xObs, xFul)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Train the model and save the best
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
epochs = 40
latent_dim = 1024
StreamAE = AutoEncoder(latent_dim)
Best = 1.0e16

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for (trainObs_x, trainFul_x) in zip(trainObs_dataset, trainFul_dataset):
        train_step(StreamAE, trainObs_x, trainFul_x, optimizer)
    end_time = time.time()

    totalLoss = tf.keras.metrics.Mean()
    reconstrLoss = tf.keras.metrics.Mean()
    ssimLoss = tf.keras.metrics.Mean()
    for (testObs_x, testFul_x) in zip(testObs_dataset, testFul_dataset):
        (total, reconstr, ss) = compute_loss2(StreamAE, testObs_x, testFul_x)
        totalLoss(total)
        reconstrLoss(reconstr)
        ssimLoss(ss)
    Current = totalLoss.result()
    print("Epoch: {}".format(epoch))
    print("  Total Loss:          {:12.6f}".format(Current))
    print("  Reconstruction Loss: {:12.6f}".format(reconstrLoss.result()))
    print("  MS-SSIM Loss:        {:12.6f}".format(ssimLoss.result()))
    print("  Time elapse:         {:12.6f}".format(end_time - start_time))

    if Current < Best:
        Best = Current
        os.system("rm -rf StreamAutoEncoder_L1024")
        StreamAE.save("StreamAutoEncoder_L1024")
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        print("+++ New model saved")
    else:
        print("--- Best loss so far: {}".format(Best))


tf.keras.backend.set_value(optimizer.learning_rate, 5.0e-5)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for (trainObs_x, trainFul_x) in zip(trainObs_dataset, trainFul_dataset):
        train_step(StreamAE, trainObs_x, trainFul_x, optimizer)
    end_time = time.time()

    totalLoss = tf.keras.metrics.Mean()
    reconstrLoss = tf.keras.metrics.Mean()
    ssimLoss = tf.keras.metrics.Mean()
    for (testObs_x, testFul_x) in zip(testObs_dataset, testFul_dataset):
        (total, reconstr, ss) = compute_loss2(StreamAE, testObs_x, testFul_x)
        totalLoss(total)
        reconstrLoss(reconstr)
        ssimLoss(ss)
    Current = totalLoss.result()
    print("Epoch: {}".format(epoch + 40))
    print("  Total Loss:          {:12.6f}".format(Current))
    print("  Reconstruction Loss: {:12.6f}".format(reconstrLoss.result()))
    print("  MS-SSIM Loss:        {:12.6f}".format(ssimLoss.result()))
    print("  Time elapse:         {:12.6f}".format(end_time - start_time))

    if Current < Best:
        Best = Current
        os.system("rm -rf StreamAutoEncoder_L1024")
        StreamAE.save("StreamAutoEncoder_L1024")
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        print("+++ New model saved")
    else:
        print("--- Best loss so far: {}".format(Best))


tf.keras.backend.set_value(optimizer.learning_rate, 2.5e-5)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for (trainObs_x, trainFul_x) in zip(trainObs_dataset, trainFul_dataset):
        train_step(StreamAE, trainObs_x, trainFul_x, optimizer)
    end_time = time.time()

    totalLoss = tf.keras.metrics.Mean()
    reconstrLoss = tf.keras.metrics.Mean()
    ssimLoss = tf.keras.metrics.Mean()
    for (testObs_x, testFul_x) in zip(testObs_dataset, testFul_dataset):
        (total, reconstr, ss) = compute_loss2(StreamAE, testObs_x, testFul_x)
        totalLoss(total)
        reconstrLoss(reconstr)
        ssimLoss(ss)
    Current = totalLoss.result()
    print("Epoch: {}".format(epoch + 80))
    print("  Total Loss:          {:12.6f}".format(Current))
    print("  Reconstruction Loss: {:12.6f}".format(reconstrLoss.result()))
    print("  MS-SSIM Loss:        {:12.6f}".format(ssimLoss.result()))
    print("  Time elapse:         {:12.6f}".format(end_time - start_time))

    if Current < Best:
        Best = Current
        os.system("rm -rf StreamAutoEncoder_L1024")
        StreamAE.save("StreamAutoEncoder_L1024")
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        print("+++ New model saved")
    else:
        print("--- Best loss so far: {}".format(Best))


tf.keras.backend.set_value(optimizer.learning_rate, 1.0e-5)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for (trainObs_x, trainFul_x) in zip(trainObs_dataset, trainFul_dataset):
        train_step(StreamAE, trainObs_x, trainFul_x, optimizer)
    end_time = time.time()

    totalLoss = tf.keras.metrics.Mean()
    reconstrLoss = tf.keras.metrics.Mean()
    ssimLoss = tf.keras.metrics.Mean()
    for (testObs_x, testFul_x) in zip(testObs_dataset, testFul_dataset):
        (total, reconstr, ss) = compute_loss2(StreamAE, testObs_x, testFul_x)
        totalLoss(total)
        reconstrLoss(reconstr)
        ssimLoss(ss)
    Current = totalLoss.result()
    print("Epoch: {}".format(epoch + 120))
    print("  Total Loss:          {:12.6f}".format(Current))
    print("  Reconstruction Loss: {:12.6f}".format(reconstrLoss.result()))
    print("  MS-SSIM Loss:        {:12.6f}".format(ssimLoss.result()))
    print("  Time elapse:         {:12.6f}".format(end_time - start_time))

    if Current < Best:
        Best = Current
        os.system("rm -rf StreamAutoEncoder_L1024")
        StreamAE.save("StreamAutoEncoder_L1024")
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        print("+++ New model saved")
    else:
        print("--- Best loss so far: {}".format(Best))
