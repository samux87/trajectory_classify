from keras.preprocessing import sequence
from keras.layers import *
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import config
from sklearn.metrics.classification import accuracy_score, confusion_matrix, precision_recall_fscore_support
import os

# from tensorflow.python.framework.

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'netooarest'
plt.rcParams['image.cmap'] = 'gray'

def evaluation(Y_test, Y_predict):
    # Need
    print("Overall Accuracy Score: {}".format(accuracy_score(Y_test, Y_predict)))
    print("Confusion Matrix: ")
    print(confusion_matrix(Y_test, Y_predict))
    print("Precision, Recall, F-socre, suppport: ")
    print(precision_recall_fscore_support(Y_test, Y_predict))
    return accuracy_score(Y_test, Y_predict), precision_recall_fscore_support(Y_test, Y_predict)[1][1]

def loadSeries(filelist,idx):
    filename = filelist[idx]
    name = filename.split('/')[-1].strip('.mat')
    label = filename.split('/')[-2]
    y = label
    data = open(filename, 'r').readlines()[3].replace('\n','').split(':')[1].split(',')
    series = [int(d) for d in data]
    return name, label, MBMC14list.index(y), series

def attention(args):
    batch_len_hiddenstates, lengths = args
    hidden_size = K.shape(batch_len_hiddenstates)[2]
    out_tensor = None
    for i, len in enumerate(lengths):
        content = K.slice(batch_len_hiddenstates , [i, 0, 0] , [1, len, hidden_size])
        query = K.slice(batch_len_hiddenstates, [i, len-1, 0] , [1, 1, hidden_size])
        print(K.shape(content), K.shape(query))
        score = K.dot(query, content)
        print(K.shape(score))
        prob = K.softmax(score, axis = -1)
        atten_vec = K.dot(prob, content)
        print(K.shape(atten_vec))
        if out_tensor == None:
            out_tensor = atten_vec
        else:
            out_tensor = K.stack([out_tensor, atten_vec], axis=0)
    print(K.shape(out_tensor))
    return out_tensor



# datasetDir = r'/media/ubuntu/Storage/Proj/trajs/allData/seriesFeatResampled'
datasetDir = r'/home/ubuntu/Documents/traj_mining/data/categories14_allfea_30'

MBMC14list = []

fileList = []
for root,dirs,files in os.walk(datasetDir):
    for file in files:
        fileList.append(os.path.join(root,file))

fileList.sort()

from sklearn.model_selection import train_test_split

traindata, testdata = train_test_split(fileList,test_size=0.2, random_state = 0)

# traindata  = traindata[:5000]
# testdata = testdata[:2000]
fileList = []
for root, dirs, files in os.walk(datasetDir):
    for file in files:
        fileList.append(os.path.join(root, file))

fileList.sort()

traindata, testdata = train_test_split(fileList, test_size=0.2, random_state=0)

# Train Test Split
# traindata  = traindata[:5123]
# testdata = testdata[:2345]
xs = []
ys = []
trainlen = []
for i, file in enumerate(traindata):
    if i % 1000 == 0:
        print(i)
    _, _, y, x = loadSeries(traindata, i)
    if len(x) > config.min_grid_len:
        if (len(x) > 300):
            trainlen.append(300)
            xs.append(x[:300])
        else:
            trainlen.append(len(x))
            xs.append(x)
        ys.append(y)
xs = sequence.pad_sequences(xs, maxlen=300, padding='post')
print(xs.shape)
z = sequence.pad_sequences(xs, maxlen=300, padding='post')
ys_ = to_categorical(ys)
print(z.shape)
print(ys_.shape)

testxs = []
testys = []
testlen = []
for i, file in enumerate(testdata):
    _, _, y, x = loadSeries(testdata, i)
    if len(x) > config.min_grid_len:
        if (len(x) > 300):
            testlen.append(300)
            testxs.append(x[:300])
        else:
            testlen.append(len(x))
            testxs.append(x)
        testys.append(y)
    if i % 1000 == 0:
        print(i)
testxs = sequence.pad_sequences(testxs, maxlen=300, padding='post')
testz = sequence.pad_sequences(testxs, maxlen=300, padding='post')
testys_ = to_categorical(testys)
print(testz.shape)
print(testys_.shape)


inputs = Input(shape=(300,))
embedding = Embedding(170000, 64)
lstm = Sequential()
lstm.add(Bidirectional(LSTM(128,dropout=0,return_sequences=True,
              kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', implementation=1,
            recurrent_regularizer='l2', bias_regularizer='l2')))
# lstm.add(Bidirectional(LSTM(64,dropout=0,return_sequences=True,
#               kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros', implementation=1,
#             recurrent_regularizer='l2', bias_regularizer='l2')))
# lstm.add(Bidirectional(LSTM(32,dropout=0,return_sequences=True,
#               kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros', implementation=1,
#             recurrent_regularizer='l2', bias_regularizer='l2')))
# lstm.add(LSTM(64, dropout=0, return_sequences=True,
#                kernel_initializer='glorot_uniform',
#                recurrent_initializer='orthogonal',
#                bias_initializer='zeros', implementation=1,
#                recurrent_regularizer='l2', bias_regularizer='l2'))
flatten = Flatten()
dense1 = Dense(256,activation='relu')
dense2 = Dense(14,activation='softmax')

embedding_out = embedding(inputs)
states_out = lstm(embedding_out)
print(embedding_out.shape, states_out.shape)

# atten_out = Lambda(attention, output_shape=(64,))(states_out)
out = dense2(dense1(flatten(states_out)))

model = Model([inputs], out)
import numpy as np
import pandas as pd
import math, pickle

import keras.layers.recurrent
# model = Sequential()
# model.add(Embedding(170000,64))
# # model.add(LSTM(64,dropout_W=0.5,dropout_U=0.5,return_sequences=True))
# model.add(LSTM(64,dropout=0,return_sequences=True,
#               kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros'))
# model.add(LSTM(64,dropout=0,return_sequences=True,
#               kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros'))
#
# model.add(Flatten())
# model.add(Dense(256,activation='relu'))-
# model.add(Dense(14,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'] )
modelResult = model.fit(z,ys_,epochs=config.training_epoch,
                        validation_data=(testz,testys_),
                        batch_size=config.batch_size,
                        verbose=1)

testpredscore = model.predict(testz)

testpredy = np.argmax(testpredscore,axis=1)

evaluation(testys, testpredy)

print ((testpredy==np.array(testys)).sum() / len(testys))