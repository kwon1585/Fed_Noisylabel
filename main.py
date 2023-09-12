#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import random
import numpy as np
import tensorflow.keras.datasets as datasets


# In[2]:


def MNIST_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    return x_train, y_train


# In[3]:


def data_to_noisy(labels, rate):
    data_cnt = len(labels)
    noisy_cnt = int(rate*data_cnt)
    
    noisy_index = random.sample(range(0, data_cnt), noisy_cnt)
    for index in noisy_index:
        now_label = labels[index]
        new_label = random.randint(0,9)
        while (new_label == now_label):
            new_label = random.randint(0,9)
        labels[index] = new_label
    
    noisy_index.sort()
    
    return labels, noisy_index


# In[4]:


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out


# In[9]:


epochs = 8
batch_size = 1
noisy_rate = 0.3

true_loss = []
noisy_loss = []
true_loss_10 = []
noisy_loss_10 = []

true_labels = np.eye(10)
X, Y = MNIST_data()
Y, noisy_index = data_to_noisy(Y, noisy_rate)

model = CNN()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

cnt = 0
noisy_index_cp = noisy_index
noisy = noisy_index.pop(0)

model.train()
for x_train, y_train in zip(X, Y):
    x_train = torch.Tensor(x_train.reshape(batch_size, 1, 28, 28))
    y_train = torch.Tensor(true_labels[y_train].reshape(batch_size, 10))

    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if cnt == noisy:
        noisy_loss.append(loss)
        if len(noisy_index):
            noisy = noisy_index.pop(0)
    else: 
        true_loss.append(loss.item())

    cnt += 1

#============================================================================
cnt = 0
noisy_index = noisy_index_cp
noisy = noisy_index.pop(0)
for epoch in range(epochs):
    for x_train, y_train in zip(X, Y):
        x_train = torch.Tensor(x_train.reshape(batch_size, 1, 28, 28))
        y_train = torch.Tensor(true_labels[y_train].reshape(batch_size, 10))

        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
#============================================================================
cnt = 0
noisy_index = noisy_index_cp
noisy = noisy_index.pop(0)
for x_train, y_train in zip(X, Y):
    x_train = torch.Tensor(x_train.reshape(batch_size, 1, 28, 28))
    y_train = torch.Tensor(true_labels[y_train].reshape(batch_size, 10))

    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if cnt == noisy:
        noisy_loss_10.append(loss)
        if len(noisy_index):
            noisy = noisy_index.pop(0)
    else: 
        true_loss_10.append(loss.item())

    cnt += 1


# In[ ]:




