import numpy as np
import scipy as sp
import scipy.io
import os.path
import time
from ClassificiationRNN import ClassificationRNN
import torch
import config
from torch.optim.adam import Adam
from sklearn.metrics.classification import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


# matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


MBMC14list = []


# datasetDir = r'/media/ubuntu/Storage/Proj/trajs/allData/seriesFeatResampled'

datasetDir = config.datasetDir
def loadSeries(filelist, idx):
    filename = filelist[idx]
    name = filename.split('/')[-1].strip('.mat')
    label = filename.split('/')[-2]
    y = label
    data = open(filename, 'r').readlines()[3].replace('\n','').split(':')[1].split(',')
    series = [int(d) for d in data]
    return name, label, MBMC14list.index(y), series

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def evaluation(Y_test, Y_predict):
    # Need
    print("Overall Accuracy Score: {}".format(accuracy_score(Y_test, Y_predict)))
    print("Confusion Matrix: ")
    print(confusion_matrix(Y_test, Y_predict))
    print("Precision, Recall, F-socre, suppport: ")
    print(precision_recall_fscore_support(Y_test, Y_predict))
    return accuracy_score(Y_test, Y_predict), precision_recall_fscore_support(Y_test, Y_predict)[1][1]

def get_batch(xs, trainlen, ys, batchsize = config.batch_size):
    count_batch, j = 0, 0
    while (j+batchsize) < len(xs):
        count_batch +=1
        xs_batch = xs[j: j+batchsize][:][:]
        ys_batch = ys[j: j+batchsize]
        lengths = trainlen[j:j+batchsize]
        j = j+batchsize
        yield count_batch, xs_batch, lengths, ys_batch
    if j  < len(xs):
        count_batch +=1
        xs_batch = xs[j:][:][:]
        ys_batch = ys[j:]
        lengths = trainlen[j:]
        yield count_batch, xs_batch, lengths, ys_batch


    # for batch, lengths in zip(xs.split(batchsize), trainlen.split(batchsize)):
    #     count_batch +=1
    #     yield count_batch, xs_batch, lengths

if __name__ == '__main__':
    # Load data
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
    for i,file in enumerate(traindata):
        if i%100==0:
            print(i)
        _,_,y,x = loadSeries(traindata,i)
        if len(x) > config.min_grid_len:
            if (len(x) > 300):
                trainlen.append(300)
                xs.append(x[:300])
            else:
                trainlen.append(len(x))
                xs.append(x)
            ys.append(y)
    xs =  pad_sequences(xs, maxlen=300, padding='post')
    print(xs.shape)
    z = pad_sequences(xs, maxlen=300, padding='post')
    ys_ = to_categorical(ys)
    print (z.shape)
    print (ys_.shape)

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
        if i % 100 == 0:
            print(i)
    testxs = pad_sequences(testxs, maxlen=300, padding='post')
    testz = pad_sequences(testxs, maxlen=300, padding='post')
    testys_ = to_categorical(testys)
    print(testz.shape)
    print (testys_.shape)

    # Train
    model = ClassificationRNN(place_dim = 180000,
                               isAttention=config.isAttention)
    optimizer = Adam(model.parameters(), lr= 0.001)
    celoss = torch.nn.CrossEntropyLoss()

    max_acc = 0
    prev_model = None
    for epoch in range(config.training_epoch):
        print("-------------------------")
        model.train()
        start = time.time()
        for j, xs_batch, trainlen_batch, ys_batch in get_batch(xs, trainlen, ys):
            # print(len(xs), len(trainlen),len(trainlen_batch))
            last_out = model(xs_batch.astype(int), trainlen_batch)
            # target = torch.LongTensor(ys_batch).cuda()
            # print(ys_batch)
            target = torch.LongTensor(ys_batch).cuda()
            loss = celoss(last_out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_end = time.time()
            if (j + 1) % config.print_batch == 0:
                result = [list(i).index(i.max()) for i in last_out.cpu().detach().numpy()]
                train_acc = accuracy_score(ys_batch, result)
                print('Epoch [{}/{}], Step [{}/{}], Total_Loss: {}, Train Acc: {}\t' \
                      'Time_cost: {}'. \
                      format(epoch + 1, config.training_epoch, j + 1, int(len(xs) / config.batch_size) + 1,
                             loss.data[0], train_acc, batch_end - start))
        model.eval()
        all_out = []
        for j, testxs_batch, testlen_batch, testys_batch in get_batch(testxs, testlen, testys):
            last_out = model(testxs_batch.astype(int), testlen_batch)
            all_out.append(last_out)
        print(len(all_out),all_out[-2].size(), all_out[-1].size())
        all_out = torch.cat(all_out, dim = 0)
        print (all_out.size())
        target = torch.LongTensor(testys).cuda()
        result = [list(i).index(i.max()) for i in all_out.cpu().detach().numpy()]
        acc = accuracy_score(testys, result)
        print("Overall Accuracy Score: {}".format(acc))

        print (prev_model)
        if max_acc< acc:
            save_model_name = './trained_models/{}_isAttention_{}_BiDir_{}_layer_{}_acc_{}'\
                .format(datasetDir.split('/')[-1], config.isAttention, config.biDirectional, config.number_layers, acc)
            max_acc = acc
            print(save_model_name)
            time.sleep(1)
            torch.save(model.cpu().state_dict(), save_model_name)
            if prev_model == None:
                prev_model = save_model_name
            else:
                os.remove(prev_model)
                prev_model = save_model_name
            model.cuda()