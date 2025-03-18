#!/usr/bin/env python

# Classes to create the data for training model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os, glob
import cmath
import re
import sys
import io
import math


import scipy.io as scio


class DataProcessor:
    def __init__(self, path, seq_len, channels):
        self.training_path = path
        self.sequence_length = seq_len
        self.channels = channels

    def get_dataset_from_path(self, buffer):
        read_data = tf.data.Dataset.list_files(self.training_path)
        dataset = read_data.repeat().shuffle(buffer_size=buffer)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        return dataset

    def provide_video_data(self, buffer, batch_size, height, width):
        '''
        :return: tf dataset
        '''
        def read_tfrecord(serialized_example):
            features = {'x': tf.io.FixedLenFeature([height * width * self.sequence_length * self.channels, ],
                                                   dtype=tf.float32)}
            example = tf.io.parse_single_example(serialized_example, features)
            return example['x']

        dataset = self.get_dataset_from_path(buffer)
        dataset = dataset.map(read_tfrecord, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        return dataset


class AROne:
    '''
    :param D: dimension of x
    :param T: sequence length
    :param phi: parameters for AR model
    :param s: parameter that controls the magnitude of covariance matrix
    '''
    def __init__(self, D, T, phi, s, burn=10):
        self.D = D
        self.T = T
        self.phi = phi
        self.Sig = np.eye(D) * (1 - s) + s
        self.chol = np.linalg.cholesky(self.Sig)
        self.burn = burn

    def batch(self, N):
        x0 = np.random.randn(N, self.D)
        x = np.zeros((self.T + self.burn, N, self.D))
        x[0, :, :] = x0
        for i in range(1, self.T + self.burn):
            x[i, ...] = self.phi * x[i - 1] + np.random.randn(N, self.D) @ self.chol.T

        x = x[-self.T:, :, :]
        x = np.swapaxes(x, 0, 1)
        return x.astype("float32")


class Gaussian:
    def __init__(self, D=1):
        self.D = D

    def batch(self, batch_size):
        return np.random.randn(batch_size, 1, self.D)


class SineImage(object):
    '''
    :param Dx: dimensionality of of data at each time step
    :param angle: rotation
    :param z0: initial position and velocity
    :param rand_std: gaussian randomness in the latent trajectory
    :param noise_std: observation noise at output
    '''
    def __init__(self, Dx=20, angle=np.pi / 6., z0=None, rand_std=0.0, noise_std=0.0, length=None, amp=1.0):
        super().__init__()
        self.D = 2
        self.Dx = Dx
        self.z0 = z0

        self.A = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.rand_std = rand_std
        self.noise_std = noise_std
        self.length = length
        self.amp = amp

    def sample(self, n, T):
        # n: number of samples
        # T: lenght of each sample
        if self.z0 is None:
            z = np.random.randn(n, 2)
            z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        else:
            z = np.tile(self.z0, (n, 1))

        zs = []
        for t in np.arange(T):
            m = self.conditional_param(z)
            z = m + np.random.randn(*m.shape) * self.rand_std
            zs += z,

        zs = np.stack(zs, 1)

        grid = np.linspace(-1.5, 1.5, self.Dx)

        mean = np.exp(- 0.5 * (zs[..., :1] - grid) ** 2 / 0.3 ** 2) * self.amp
        mean = mean.reshape(n, -1)
        xs = mean + np.random.randn(*mean.shape) * self.noise_std

        return zs, xs.reshape(n, T, self.Dx)

    def conditional_param(self, zt):

        slope = 1.0
        r = np.sqrt(np.sum(zt ** 2, -1))
        r_ratio = 1.0 / (np.exp(-slope * 4 * (r - 0.3)) + 1) / r

        ztp1 = zt @ self.A
        ztp1 *= r_ratio[..., None]

        return ztp1

    def batch(self, batch_size):
        return self.sample(batch_size, self.length)[1]


class NPData(object):
    def __init__(self, data, batch_size, nepoch=np.inf, tensor=True):
        self.data = data
        self.N, self.length = data.shape[0:2]
        self.epoch = 0
        self.counter = 0
        np.random.shuffle(self.data)
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.tensor = tensor

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.N / self.batch_size))

    def __next__(self):
        if (self.counter + 1) * self.batch_size > self.N:
            self.epoch += 1
            np.random.shuffle(self.data)
            self.counter = 0

        if np.isfinite(self.nepoch) and self.epoch == self.nepoch:
            raise StopIteration

        idx = slice(self.counter * self.batch_size, (self.counter + 1) * self.batch_size)
        batch = self.data[idx]
        self.counter += 1
        if self.tensor:
            batch = tf.cast(batch, tf.float32)
        return batch

    def batch(self, batch_size):
        return self.__next__()


class EEGData(NPData):
    '''
    :param Dx: dimensionality of of data at each time step
    :param length: sequence length
    :param batch size: batch size
    '''

    def __init__(self, Dx, length, batch_size, nepoch=np.inf, tensor=True, seed=0, prefix="", downsample=1):
        # nsubject x n trials x channel x times_steps
        # all_data = np.load(prefix + "data/eeg/eeg_data.npy", allow_pickle=True)
        # train_data = []
        # test_data = []
        # sep = 0.75
        # np.random.RandomState(seed).shuffle(all_data)
        # for sub_data in all_data:
        #     ntrial = int(sep * len(sub_data))
        #     train_data += sub_data[:ntrial, :downsample * length:downsample, :Dx],
        #     test_data += sub_data[ntrial:, :downsample * length:downsample, :Dx],
        #     assert train_data[-1].shape[1] == length
        #     assert train_data[-1].shape[2] == Dx

        # self.train_data = self.normalize(train_data)
        # self.test_data = self.normalize(test_data)
        # self.all_data = np.concatenate([self.train_data, self.test_data], 0)
        # super().__init__(self.train_data, batch_size, nepoch, tensor)

        files = {
            'MFP': ['Training.mat']
        }

        filename = "MFP"
        file = files[filename]
        filepath = "./data/CVACaseStudy/"

        if filename == "MFP":
            data = scio.loadmat(filepath + file[0])
            data_t1 = data["T1"]
            data_t2 = data["T2"]
            data_t3 = data["T3"]

        print(data_t1.shape, data_t2.shape, data_t3.shape)



        # Split the data into training, validation and test sets: 0,6, 0.2, 0.2
        train_ratio = 0.6
        valid_ratio = 0.2
        test_ratio = 1 - train_ratio - valid_ratio 

        # full data

        new_data_train_t1 = data_t1[:int(data_t1.shape[0] * train_ratio)] 
        new_data_train_t2 = data_t2[:int(data_t2.shape[0] * train_ratio)]
        new_data_train_t3 = data_t3[:int(data_t3.shape[0] * train_ratio)]

        new_data_valid_t1 = data_t1[int(data_t1.shape[0] * train_ratio) : int(data_t1.shape[0] * (train_ratio+valid_ratio))] 
        new_data_valid_t2 = data_t2[int(data_t2.shape[0] * train_ratio) : int(data_t2.shape[0] * (train_ratio+valid_ratio))]
        new_data_valid_t3 = data_t3[int(data_t3.shape[0] * train_ratio) : int(data_t3.shape[0] * (train_ratio+valid_ratio))]

        new_data_test_t1 = data_t1[int(data_t1.shape[0] * (train_ratio+valid_ratio)) : data_t1.shape[0]] 
        new_data_test_t2 = data_t2[int(data_t2.shape[0] * (train_ratio+valid_ratio)) : data_t2.shape[0]]
        new_data_test_t3 = data_t3[int(data_t3.shape[0] * (train_ratio+valid_ratio)) : data_t3.shape[0]]

        # Combine the 3 datasets into each of the training, validation and testing sets
        training_set = np.vstack((new_data_train_t1,new_data_train_t2,new_data_train_t3))
        validation_set = np.vstack((new_data_valid_t1,new_data_valid_t2,new_data_valid_t3))
        testing_set = np.vstack((new_data_test_t1,new_data_test_t2,new_data_test_t3))


        idx_var_remove = list(set([-1])) # remove output_vars and last variable, ensure that both are different with set()
        pressure_var = 5 # 4th index

        X_train = np.delete(training_set, idx_var_remove, axis=1)
        Y_train = training_set[:, pressure_var - 1]

        X_valid = np.delete(validation_set, idx_var_remove, axis=1)
        Y_valid = validation_set[:, pressure_var -1]

        X_test = np.delete(testing_set, idx_var_remove, axis=1)
        Y_test = testing_set[:, pressure_var -1]

        print(X_train.shape)


        def get_norm_param(X,Y):
            x = X
            y = Y

            keys = ['x_min','x_max','y_min','y_max','x_mean','x_std','y_mean','y_std']
            norm_param = {}
            for key in keys: norm_param[key] = []
            
            norm_param['x_min']  = np.min(x, axis=0)
            norm_param['x_max']  = np.max(x, axis=0)
            norm_param['y_min']  = np.min(y, axis=0)
            norm_param['y_max']  = np.max(y, axis=0)
            norm_param['x_mean'] = np.mean(x, axis=0)
            norm_param['x_std']  = np.std(x, axis=0)
            norm_param['y_mean'] = np.mean(y, axis=0)
            norm_param['y_std']  = np.std(y, axis=0)

            return norm_param

        norm_param_train = get_norm_param(X_train,Y_train)

        def normalize(X,Y,norm_param,method):
            x = X
            y = Y

            if method == 'minmax':
                X_norm = (x - norm_param['x_min']) / (norm_param['x_max'] - norm_param['x_min'])
                Y_norm = (y - norm_param['y_min']) / (norm_param['y_max'] - norm_param['y_min'])


            elif method == 'standardize':
                X_norm = (x - norm_param['x_mean']) / norm_param['x_std']
                Y_norm = (y - norm_param['y_mean']) / norm_param['y_std']

            else:
                raise TypeError("Normalization Method Not Known")

            return X_norm, Y_norm

        norm_method = "minmax"
        X_train_norm, Y_train_norm = normalize(X_train,Y_train,norm_param_train,norm_method)
        X_valid_norm, Y_valid_norm = normalize(X_valid,Y_valid,norm_param_train,norm_method)
        X_test_norm, Y_test_norm = normalize(X_test,Y_test,norm_param_train,norm_method)





        import torch 
        from torch.utils.data import Dataset,DataLoader

        class MyDataset(Dataset):
            def __init__(self, X, Y, his_length, pred_length, pred_mode):
                self.x = X
                self.y = Y
                self.his_length = his_length 
                self.pred_length = pred_length
                self.mode = pred_mode

            def __getitem__(self,index):

                x = self.x[index : index + self.his_length]
                # print(x.shape)

                if self.mode == 'current_step':
                    y = self.y[index + self.his_length - 1]
                    y = torch.Tensor([y])

                elif self.mode == 'multi_step':
                    y = self.y[index + self.his_length : index + self.his_length + self.pred_length]
                    y = torch.Tensor(y)

                else:
                    raise TypeError('Prediction Model is not Known')
                
                y = torch.ones_like(y)


                return torch.Tensor(x)#, y
            
            def __len__(self):
                if self.mode == 'current_step':
                    # print(len(self.x) - self.his_length + 1)
                    return len(self.x) - self.his_length + 1

                elif self.mode == 'multi_step':
                    return len(self.x) - self.his_length - self.pred_length + 1
                
                else: 
                    raise TypeError('Prediction Model is not Known')


        pred_mode = 'current_step' # current_step/multi_step
        his_length = 85 # L
        pred_length = 1 

        train_dataset = MyDataset(X_train_norm, Y_train_norm, his_length, pred_length, pred_mode)
        valid_dataset = MyDataset(X_valid_norm, Y_valid_norm, his_length, pred_length, pred_mode)
        test_dataset = MyDataset(X_test_norm, Y_test_norm, his_length, pred_length, pred_mode)

        batch_size = 128

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        
        ret = []
        for item in train_dataloader:
            for i in item:
                ret.append(i)


        self.train_data = np.array(ret)
        print(self.train_data.shape)

        ret = []
        for item in test_dataloader:
            for i in item:
                ret.append(i)


        self.test_data = np.array(ret)
        print(self.test_data.shape)


        self.all_data = np.concatenate([self.train_data, self.test_data], 0)
        super().__init__(self.train_data, 128, nepoch, tensor)




def plot_batch(batch_series, iters, saved_file, axis=None):
    '''
    :param batch_series: a batch of sequence
    :param iters: current iteration
    :return: plots up to six sequences on shared axis
    '''
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    batch_size = np.shape(batch_series)[0]
    num_seq = np.minimum(len(flatui), batch_size)

    for i in range(0, num_seq):
        data = [_ for _ in enumerate(batch_series[i])]
        sns.lineplot(x=[el[0] for el in data],
                     y=[el[1] for el in data],
                     color=flatui[i % len(flatui)],
                     ax=axis)
    str = "Sample plot after {} iterations".format(iters)
    plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def display_images(x, row, col, batch_size, height, width, iters, saved_file):
    fig, axe = plt.subplots(row, col, figsize=(8, 8))

    for i in range(row):
        for j in range(col):
            axe[i][j].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width]), origin="upper",
                             cmap="gray", interpolation="nearest")
            axe[i][j].set_xticks([])
            axe[i][j].set_yticks([])
    str = "Sample plot after {} iterations".format(iters)
    # plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def display_frames(x, row, batch_size, seq_len, height, width, channels, iters, saved_file):
    fig, axe = plt.subplots(row, figsize=(8, 8))

    for i in range(row):
        if channels > 1:
            axe[i].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width * seq_len, channels]),
                          origin="upper", cmap="gray", interpolation="nearest")
        else:
            axe[i].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width * seq_len]),
                          origin="upper", cmap="gray", interpolation="nearest")
        axe[i].set_xticks([])
        axe[i].set_yticks([])
    str = "Sample plot after {} iterations".format(iters)
    # plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def check_model_summary(batch_size, seq_len, model, stateful=False):
    if stateful:
        inputs = tf.keras.Input((batch_size, seq_len))
    else:
        inputs = tf.keras.Input((batch_size, seq_len))
    outputs = model.call(inputs)

    model_build = tf.keras.Model(inputs, outputs)
    print(model_build.summary())
