import sys
import cv2
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

class GAN():
    def __init__(self, deep_convolutional):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.deep_convolutional = deep_convolutional

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        if self.deep_convolutional:
          model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim))
          model.add(Reshape((32, 32, 128)))
          model.add(UpSampling2D())
          model.add(Conv2D(128, kernel_size=3, padding="same"))
          model.add(BatchNormalization(momentum=0.8))
          model.add(Activation("relu"))
          model.add(UpSampling2D())
          model.add(Conv2D(64, kernel_size=3, padding="same"))
          model.add(BatchNormalization(momentum=0.8))
          model.add(Activation("relu"))
          model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
          model.add(Activation("tanh"))
        else:
          model.add(Dense(256, input_dim=self.latent_dim))
          model.add(LeakyReLU(alpha=0.2))
          model.add(BatchNormalization(momentum=0.8))
          model.add(Dense(512))
          model.add(LeakyReLU(alpha=0.2))
          model.add(BatchNormalization(momentum=0.8))
          model.add(Dense(1024))
          model.add(LeakyReLU(alpha=0.2))
          model.add(BatchNormalization(momentum=0.8))
          model.add(Dense(np.prod(self.img_shape), activation='tanh'))
          model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        if self.deep_convolutional:
          model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Dropout(0.25))
          model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
          model.add(ZeroPadding2D(padding=((0,1),(0,1))))
          model.add(BatchNormalization(momentum=0.8))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Dropout(0.25))
          model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
          model.add(BatchNormalization(momentum=0.8))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Dropout(0.25))
          model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
          model.add(BatchNormalization(momentum=0.8))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Dropout(0.25))
          model.add(Flatten())
          model.add(Dense(1, activation='sigmoid'))
        else:
          model.add(Flatten(input_shape=self.img_shape))
          model.add(Dense(512))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Dense(256))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Dense(1, activation='sigmoid'))
          
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def get_batik_dataset(self):
        DATASET_FOLDER = "Batik300"
        SHAPE = (128, 128,)

        dataset = []

        file_names = os.listdir(DATASET_FOLDER)
        for file_name in file_names:
            image = cv2.imread(DATASET_FOLDER + "/" + file_name)
            try:
              image = cv2.resize(image, SHAPE)
              dataset.append(image)
            except:
              print(file_name)  

        return np.array(dataset)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = self.get_batik_dataset()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        if self.deep_convolutional:
          fig.savefig("result/deep_convolutional_%d.png" % epoch)
        else:
          fig.savefig("result/%d.png" % epoch)
        plt.close()

start = time.time()
gan = GAN(deep_convolutional=False)
gan.train(epochs=4500, batch_size=32, sample_interval=200)
end = time.time()
print("Elapsed time: %d second" % (end - start))

start = time.time()
gan = GAN(deep_convolutional=True)
gan.train(epochs=4500, batch_size=32, sample_interval=200)
end = time.time()
print("Elapsed time: %d second" % (end - start))
