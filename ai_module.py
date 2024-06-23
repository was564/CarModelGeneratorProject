import glob
import os
import time

import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Generator:
    """
    생성기 신경망 모델
    """

    def __init__(self):
        self.z_size = 200
        self.gen_filters = [512, 256, 128, 64, 1]
        self.gen_kernel_sizes = [4, 4, 4, 4, 4]
        self.gen_strides = [1, 2, 2, 2, 2]
        self.gen_input_shape = (1, 1, 1, self.z_size)
        self.gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
        self.gen_convolutional_blocks = 5
        self.gen_model = None

    def build(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv3DTranspose(filters=self.gen_filters[0],
                                                  kernel_size=self.gen_kernel_sizes[0],
                                                  strides=self.gen_strides[0],
                                                  input_shape=self.gen_input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation=self.gen_activations[0]))

        for i in range(self.gen_convolutional_blocks - 1):
            model.add(tf.keras.layers.Conv3DTranspose(filters=self.gen_filters[i + 1],
                                                      kernel_size=self.gen_kernel_sizes[i + 1],
                                                      strides=self.gen_strides[i + 1],
                                                      padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation(activation=self.gen_activations[i + 1]))

        self.gen_model = model

    def summary(self):
        self.gen_model.summary()

    def get(self):
        return self.gen_model


# Discriminator Network
class Discriminator:
    """
    판별기 신경망 모델
    """

    def __init__(self):
        self.dis_input_shape = (64, 64, 64, 1)
        self.dis_filters = [64, 128, 256, 512, 1]
        self.dis_kernel_sizes = [4, 4, 4, 4, 4]
        self.dis_strides = [2, 2, 2, 2, 1]
        self.dis_paddings = ['same', 'same', 'same', 'same', 'valid']
        self.dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
        self.dis_convolutional_blocks = 5
        self.dis_model = None

    def build(self):

        # dis_input_layer = tf.keras.layers.InputLayer(input_shape=self.dis_input_shape)
        # a = tf.keras.layers.Conv3D(filters=self.dis_filters[0],
        #                                       kernel_size=self.dis_kernel_sizes[0],
        #                                       strides=self.dis_strides[0],
        #                                       padding=self.dis_paddings[0])(dis_input_layer)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv3D(filters=self.dis_filters[0],
                                         kernel_size=self.dis_kernel_sizes[0],
                                         strides=self.dis_strides[0],
                                         padding=self.dis_paddings[0], input_shape=self.dis_input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(self.dis_alphas[0]))

        for i in range(self.dis_convolutional_blocks - 1):
            model.add(tf.keras.layers.Conv3D(filters=self.dis_filters[i + 1],
                                             kernel_size=self.dis_kernel_sizes[i + 1],
                                             strides=self.dis_strides[i + 1],
                                             padding=self.dis_paddings[i + 1]))
            model.add(tf.keras.layers.BatchNormalization())
            if self.dis_activations[i + 1] == 'leaky_relu':
                model.add(tf.keras.layers.LeakyReLU(self.dis_alphas[i + 1]))
            elif self.dis_activations[i + 1] == 'sigmoid':
                model.add(tf.keras.layers.Activation(activation='sigmoid'))

        self.dis_model = model

    def summary(self):
        self.dis_model.summary()

    def get(self):
        return self.dis_model


# def write_log(callback, name, value, batch_no):
#     summary = tf.summary()
#     summary_value = summary.value.add()
#     summary_value.simple_value = value
#     summary_value.tag = name
#     callback.writer.add_summary(summary, batch_no)
#     callback.writer.flush()

def get3DImages(data_dir, train=True, cube_len=64, obj_ratio=1.0):
    data_dir += 'train/' if train else 'test/'
    print('data_dir', data_dir)
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    file_list = file_list[0:int(obj_ratio * len(file_list))]
    volumeBatch = np.asarray([getVoxelsFromMat(data_dir + f, cube_len) for f in file_list], dtype=np.bool)
    # all_files = np.random.choice(glob.glob(data_dir), size=10)
    # # all_files = glob.glob(data_dir)
    # all_volumes = np.asarray([getVoxelsFromMat(f) for f in all_files], dtype=np.bool)
    return volumeBatch


def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)
    plt.close()


def plotAndSaveVoxel(file_path, voxel):
    """
    Plot a voxel
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxel, edgecolor="red")
    # plt.show()
    plt.savefig(file_path)


# 3D-GAN Class
class ThreeDGAN:

    def __init__(self, object_name):
        self.data_dir = "./3DShapeNets/volumetric_data/{}/30/".format(object_name)
        self.gen_learning_rate = 0.0025
        self.dis_learning_rate = 10e-5
        self.beta = 0.5
        self.batch_size = 32
        self.generated_volumes_dir = 'generated_volumes'
        self.log_dir = 'logs'
        self.z_size = 200

        self.volumes = None

        self.generator = None
        self.discriminator = None
        self.tf_board = None

        self.adversarial_model = None

    def build(self):

        print('Discriminator build...')
        d = Discriminator()
        d.build()
        self.discriminator = d.get()
        print('Discriminator build complete')

        print('Generator build...')
        g = Generator()
        g.build()
        self.generator = g.get()
        print('Generator build complete')

    def create_models(self):
        gen_optimizer = tf.keras.optimizers.Adam(lr=self.gen_learning_rate, beta_1=self.beta)
        dis_optimizer = tf.keras.optimizers.Adam(lr=self.dis_learning_rate, beta_1=0.9)

        self.build()
        self.discriminator.compile(optimizer=dis_optimizer, loss=tf.keras.losses.BinaryCrossentropy())
        self.generator.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())

        self.discriminator.trainable = False
        print('here')
        # input_layer = tf.keras.layers.InputLayer(input_shape=(1, 1, 1, self.z_size))
        # generated_volumes = self.generator(input_layer)
        # validity = self.discriminator(generated_volumes)
        # self.adversarial_model = tf.keras.Model(inputs=[input_layer], outputs=[validity])
        # self.adversarial_model.compile(optimizer=gen_optimizer, loss=tf.keras.losses.BinaryCrossentropy())
        self.adversarial_model = tf.keras.Sequential()
        self.adversarial_model.add(self.generator)
        self.adversarial_model.add(self.discriminator)
        self.adversarial_model.compile(optimizer=gen_optimizer, loss=tf.keras.losses.BinaryCrossentropy())

        print("Loading data...")
        self.volumes = get3DImages(data_dir=self.data_dir)
        self.volumes = self.volumes[..., np.newaxis].astype(np.float)
        print("Data loaded...length of volumes: ", len(self.volumes))

        self.tf_board = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))
        self.tf_board.set_model(self.generator)
        self.tf_board.set_model(self.discriminator)

    def train(self, epochs):
        import gc
        for epoch in range(epochs):
            gc.collect()
            print('Epoch:', epoch)

            gen_losses = []
            dis_losses = []

            number_of_batches = int(self.volumes.shape[0] / self.batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)
                z_sample = np.random.normal(0, 0.33, size=[self.batch_size, 1, 1, 1, self.z_size]).astype(
                    np.float32)
                volumes_batch = self.volumes[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]

                gen_volumes = self.generator.predict(z_sample, verbose=3)

                """
                Train the discriminator network
                """
                self.discriminator.trainable = True

                labels_real = np.reshape(np.ones((self.batch_size,)), (-1, 1, 1, 1, 1))
                labels_fake = np.reshape(np.zeros((self.batch_size,)), (-1, 1, 1, 1, 1))

                loss_real = self.discriminator.train_on_batch(volumes_batch, labels_real)
                loss_fake = self.discriminator.train_on_batch(gen_volumes, labels_fake)

                d_loss = 0.5 * np.add(loss_real, loss_fake)
                print("d_loss:{}".format(d_loss))

                self.discriminator.trainable = False

                """
                Train the generator network
                """
                z = np.random.normal(0, 0.33, size=[self.batch_size, 1, 1, 1, self.z_size]).astype(np.float32)
                g_loss = self.adversarial_model.train_on_batch(z, np.reshape(np.ones((self.batch_size,)),
                                                                             (-1, 1, 1, 1, 1)))
                print("g_loss:{}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

                # Every 10th mini-batch, generate volumes and save them
                if index % 10 == 0:
                    z_sample2 = np.random.normal(0, 0.33, size=[self.batch_size, 1, 1, 1, self.z_size]).astype(
                        np.float32)
                    generated_volumes = self.generator.predict(z_sample2, verbose=3)
                    '''
                    for i, generated_volume in enumerate(generated_volumes[:5]):
                        voxels = np.squeeze(generated_volume)
                        voxels[voxels < 0.5] = 0.
                        voxels[voxels >= 0.5] = 1.
                        saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))
                    '''
            """
            finish one Epoch step
            """
            tf.summary.scalar('g_loss', np.mean(gen_losses), step=epoch)
            tf.summary.scalar('d_loss', np.mean(dis_losses), step=epoch)
            # write_log(self.tf_board, 'g_loss', np.mean(gen_losses), epoch)
            # write_log(self.tf_board, 'd_loss', np.mean(dis_losses), epoch)
            self.generator.save_weights(os.path.join("models", f"generator_weights-{epoch}.h5"))
            self.discriminator.save_weights(os.path.join("models", f"discriminator_weights-{epoch}.h5"))
        """
        finish all Epochs & Save Models
        """
        self.generator.save_weights(os.path.join("models", "generator_weights.h5"))
        self.discriminator.save_weights(os.path.join("models", "discriminator_weights.h5"))

    def load(self, epoch=None):
        self.build()
        # Load model weights
        if epoch is not None:
            self.generator.load_weights(os.path.join("models", f"generator_weights-{epoch}.h5"), True)
            self.discriminator.load_weights(os.path.join("models", f"discriminator_weights-{epoch}.h5"), True)
        else:
            self.generator.load_weights(os.path.join("models", "generator_weights.h5"), True)
            self.discriminator.load_weights(os.path.join("models", "discriminator_weights.h5"), True)

    def predict(self):
        z_sample = np.random.normal(0, 1, size=[self.batch_size, 1, 1, 1, self.z_size]).astype(np.float32)
        generated_volumes = self.generator.predict(z_sample, verbose=3)

        for i, generated_volume in enumerate(generated_volumes[:2]):
            voxels = np.squeeze(generated_volume)
            voxels[voxels < 0.5] = 0.
            voxels[voxels >= 0.5] = 1.
            saveFromVoxels(voxels, "results/gen_{}".format(i))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.set_aspect('equal')
        ax.voxels(voxels, edgecolor='red')
        #plt.show()


def ai_main():
    # Boxel화 작업 테스트 : Airplane
    test_image_airplane = '3DShapeNets/volumetric_data/chair/30/test/chair_000000000_7.mat'
    voxels = io.loadmat(test_image_airplane)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

    # 3D Boxel 시각화
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_aspect('equal')
    ax.voxels(voxels, edgecolor='red')
    plt.show()

    object_name = 'chair'
    epochs = 3
    td_gan = ThreeDGAN(object_name)
    td_gan.create_models()

    train_summary_writer = tf.summary.create_file_writer('/logs')

    with train_summary_writer.as_default():
        td_gan.train(epochs)

    td_gan.predict()


def result_print():
    td_gan = ThreeDGAN('chair')
    td_gan.load(0)
    td_gan.predict()


if __name__ == '__main__':
    result_print()
    #ai_main()
