import time
import numpy as np
import json
import threading
import os

import tensorflow as tf
from keras.models import Model, clone_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.initializers import random_normal

# GPU memory growth setting for TF 2.x
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class RlModel:
    def __init__(self, weights_path, train_conv_layers):
        self.__angle_values = [-1, -0.5, 0, 0.5, 1]
        self.__nb_actions = 5
        self.__gamma = 0.99

        # Define model architecture
        activation = 'relu'
        pic_input = Input(shape=(59, 255, 3))
        x = Conv2D(16, (3, 3), padding='same', activation=activation, trainable=train_conv_layers, name='convolution0')(pic_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation=activation, trainable=train_conv_layers, name='convolution1')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation=activation, trainable=train_conv_layers, name='convolution2')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, kernel_initializer=random_normal(stddev=0.01), name='rl_dense')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.__nb_actions, kernel_initializer=random_normal(stddev=0.01), name='rl_output')(x)

        self.__action_model = Model(inputs=pic_input, outputs=output)
        self.__action_model.compile(optimizer=Adam(), loss='mean_squared_error')
        self.__action_model.summary()

        if weights_path:
            print('Loading weights from:', weights_path)
            self.__action_model.load_weights(weights_path, by_name=True)
        else:
            print('Not loading weights')

        self.__target_model = clone_model(self.__action_model)
        self.__model_lock = threading.Lock()

    def from_packet(self, packet):
        self.__action_model.set_weights([np.array(w) for w in packet['action_model']])
        if 'target_model' in packet:
            self.__target_model.set_weights([np.array(w) for w in packet['target_model']])

    def to_packet(self, get_target=True):
        packet = {'action_model': [w.tolist() for w in self.__action_model.get_weights()]}
        if get_target:
            packet['target_model'] = [w.tolist() for w in self.__target_model.get_weights()]
        return packet

    def update_with_gradient(self, gradients, should_update_critic):
        weights = self.__action_model.get_weights()
        for i in range(len(weights)):
            weights[i] += gradients[i]
        self.__action_model.set_weights(weights)

        if should_update_critic:
            self.__target_model.set_weights([np.array(w, copy=True) for w in weights])

    def update_critic(self):
        self.__target_model.set_weights([np.array(w, copy=True) for w in self.__action_model.get_weights()])

    def get_gradient_update_from_batches(self, batches):
        pre_states = np.array(batches['pre_states'])[:, 3, :, :, :]
        post_states = np.array(batches['post_states'])[:, 3, :, :, :]
        rewards = np.array(batches['rewards'])
        actions = batches['actions']
        is_not_terminal = np.array(batches['is_not_terminal'])

        labels = self.__action_model.predict(pre_states, batch_size=32)
        q_futures = self.__target_model.predict(post_states, batch_size=32)
        q_targets = rewards + self.__gamma * np.max(q_futures, axis=1) * is_not_terminal

        for i, action in enumerate(actions):
            labels[i][action] = q_targets[i]

        original_weights = self.__action_model.get_weights()
        self.__action_model.fit(pre_states, labels, epochs=1, batch_size=32, verbose=1)
        new_weights = self.__action_model.get_weights()

        gradients = [new - old for new, old in zip(new_weights, original_weights)]
        return [g.tolist() for g in gradients]

    def predict_state(self, observation):
        obs = np.array(observation)[3, :, :, :].reshape(1, 59, 255, 3)
        qs = self.__action_model.predict(obs)
        action = np.argmax(qs)
        return action, qs[0][action]

    def state_to_control_signals(self, state, car_state):
        if car_state.speed > 9:
            return (self.__angle_values[state], 0, 1)
        else:
            return (self.__angle_values[state], 1, 0)

    def get_random_state(self):
        return np.random.randint(0, self.__nb_actions)