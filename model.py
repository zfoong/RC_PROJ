# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 05:12:29 2021

@author: zfoong
"""

import random
import numpy as np
from scipy import sparse
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(100)
random.seed(100)

class ESN:
    def __init__(self, K, L, N):
        
        # ESN param
        self.sucess_rate = 0.5
        self.density = 0.3
        self.sr = 0.9
        self.input_scaling = 0.1
        self.leaking_rate = 0.8
        self.x_state = np.zeros((1, N), dtype=float)
        self.W = self.init_weight(N, N)
        self.W_input = (2.0*np.random.binomial(1, self.sucess_rate, [N, K]) - 1.0)
        self.alpha_decay = 0.0001
        self.alpha = 0.001
        self.history = []
        
        # Build model
        self.readout = Sequential()        
        
        # Decoer / Readout
        self.readout.add(Dense(1000, input_shape=(N+K,), activation='tanh'))
        self.readout.add(Dense(L, activation="softmax"))
        self.readout.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        self.readout.summary()
        
    def init_weight(self, r, c):
        '''Initialize Reservoir Weight Matrix / Internal Connection'''
        # Initialize connectivity
        internal_weights = sparse.rand(r,c,density=self.density).todense()
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Normalize fitting spectral radius
        E, _ = np.linalg.eig(internal_weights)
        eigen_max = np.max(np.abs(E))
        internal_weights /= np.abs(eigen_max)/self.sr      

        return internal_weights
    
    def reset_state(self):
        '''Reset Reservoir State'''
        N, _ = self.W.shape
        self.x_state = np.zeros((1, N), dtype=float)
        
        
    def compute_z_all(self, input_all):
        '''Compute All Reservoir State and Output Embedding Within Time T'''
        N, K = self.W_input.shape
        z = np.empty((0, N+K), dtype=float)
        for t in range(len(input_all)):
            input_current = np.atleast_2d(input_all[t]) * self.input_scaling
            hidden_state = self.W.dot(self.x_state.T) + self.W_input.dot(input_current.T)
            self.x_state = (1.0 - self.leaking_rate) * self.x_state + np.tanh(hidden_state).T
            z = np.hstack((self.x_state, input_current))
        return z
    
    def compute_z(self, input_x):
        '''Compute Next Reservoir State and Output Embedding'''
        # Encoder / Reservoir
        N, K = self.W_input.shape
        z = np.empty((0, N+K), dtype=float)
        input_current = np.atleast_2d(input_x) * self.input_scaling
        
        # Update reservoir state
        hidden_state = self.W.dot(self.x_state.T) + self.W_input.dot(input_current.T)
        self.x_state = (1.0 - self.leaking_rate) * self.x_state + np.tanh(hidden_state).T
        
        # Create embedding
        z = np.hstack((self.x_state, input_current))
        return z
    
    def remember(self, z, label):
        '''Append Buffer'''
        self.memory.append((z, label))
    
    def train(self, z, y):
        '''Training readout'''
        # self.readout.fit(z, y, verbose=0)
        self.history.append(self.readout.fit(z, y, verbose=0))

    def predict(self, z):
        '''Readout Prediction'''
        return self.readout.predict(z)
        
    


    
    