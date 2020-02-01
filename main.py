import os
from keras.utils.np_utils import *
from keras.constraints import max_norm
from sklearn.preprocessing import *
from model import OC

def read_data(error=0, is_train=True):
    fi = os.path.join('data/',
                      ('d0' if error < 10 else 'd') + str(error) + ('_te.dat' if is_train else '.dat'))
    with open(fi, 'r') as fr:
        data = fr.read()
    data = np.fromstring(data, dtype=np.float32, sep='   ')
    if fi == 'data/d00.dat':
        data = data.reshape(-1, 500).T
        data = data[20:500,:]
    else:
        data = data.reshape(-1, 52)
    if is_train:
        data = data[160: ]
    return data, np.ones(data.shape[0], np.int64) * error

def gen_seq_data(target, n_samples, is_train):
    seq_data, seq_labels = [], []
    for i, t in enumerate(target):
        d, _ = read_data(t, is_train)
        data = []
        length = d.shape[0] - n_samples + 1
        for j in range(n_samples):
            data.append(d[j : j + length])
        data = np.hstack(data)
        seq_data.append(data)
        seq_labels.append(np.ones(data.shape[0], np.int64) * i)
    return np.vstack(seq_data), np.concatenate(seq_labels)

def oc_network(train_data, train_labels, constraint, dropout,
                     regularizer, n_classes, TIME_STEPS, BATCH_SIZE, EPOCHS):
    template = [[TIME_STEPS, train_data.shape[2]], [128, 64], [64, 5], [n_classes, 1]]
    model = OC(template, dropout, regularizer, constraint)
    # training
    history = model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
    return model, history

if __name__ == '__main__':
    target = list(range(1, 22))
    TIME_STEPS = 10
    BATCH_SIZE, EPOCHS = 64, 150
    n_classes = len(target)

    train_data, train_labels = gen_seq_data(target, TIME_STEPS, is_train=True)
    test_data, test_labels = gen_seq_data(target, TIME_STEPS, is_train=False)

    # Preprocess: zero mean and unit standard variance
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_data_bl = train_data.reshape(train_data.shape[0], TIME_STEPS, train_data.shape[1] // TIME_STEPS)
    train_labels_one_hot = to_categorical(train_labels, len(target))
    test_data_bl = test_data.reshape(test_data.shape[0], TIME_STEPS, test_data.shape[1] // TIME_STEPS)
    test_labels_one_hot = to_categorical(test_labels, len(target))

    regularizer = None
    constraint = max_norm(3.0, axis=0)
    dropout = 0.3

    model, history = oc_network(train_data_bl, train_labels_one_hot, constraint, dropout,
                     regularizer, n_classes, TIME_STEPS, BATCH_SIZE, EPOCHS)
    model.summary()

    # testing, and output the encoder
    yhat = model.predict(test_data_bl, verbose=0)