import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import csv
from tqdm import tqdm

# load data
rows = []
with open('Data/Train/ABCD.csv', 'r') as csv_file:
    result = csv.reader(csv_file)

    for row in tqdm(result):
        rows.append(np.uint8(row))

X = np.array(rows)
X = np.random.permutation(X)

y = X[:, 0]
X = X[:, 1:]

# chuyen sang one-hot
from scipy import sparse
def convert_labels(y, C = 4):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# xu ly du lieu de co du lieu train va test
Y = convert_labels(y)
X = X.T
N_train = 50000
X_train = X[:, :N_train]
X_test = X[:, N_train:]
Y_train = Y[:, :N_train]
Y_test = Y[:, N_train:]

y_train = y[:N_train]
y_test = y[N_train:]

from predict import softmax

def cost(Yhat, Y):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

def neural_network(X, Y, W1, W2, b1, b2, eta = 1):
    max_count = 20  # so vong lap
    N = X.shape[1]  
    minibatch_size = 50
    loss = []   # de ve loss
    # du dung minibatch gradient
    for _ in tqdm(range(max_count)):
        mix_id = np.random.permutation(N)
        i = 0
        while i < N:
            true_id = mix_id[i:i+minibatch_size]
            Xi = X[:, true_id]
            Yi = Y[:, true_id]

            # feed for ward : lan truyen xuoi
            Z1 = W1.T.dot(Xi) + b1
            A1 = np.maximum(Z1, 0)
            Z2 = W2.T.dot(A1) + b2
            Yhat = softmax(Z2)

            if i % 1000 == 0:
                loss.append(cost(Yhat, Yi))
                # print(loss[-1])
                if loss[-1] < 1e-5:
                    return W1, W2, b1, b2, loss

            # backprogapation : lan truyen nguoc
            E2 = (Yhat - Yi)/N
            dW2 = A1.dot(E2.T)
            db2 = np.sum(E2, axis = 1, keepdims=True)
            E1 = W2.dot(E2)
            E1[Z1 <= 0] = 0 # gradient of ReLU
            dW1 = Xi.dot(E1.T)
            db1 = np.sum(E1, axis = 1, keepdims=True)

            # update
            W1 += -eta*dW1
            b1 += -eta*db1
            W2 += -eta*dW2
            b2 += -eta*db2

            i += minibatch_size

    return W1, W2, b1, b2, loss

# khoi tao gia tri
d0 = X_train.shape[0]
d1 = 100
d2 = 4
W1 = 0.01*np.random.randn(d0, d1)
W2 = 0.01*np.random.randn(d1, d2)
b1 = 0.01*np.random.randn(d1, 1)
b2 = 0.01*np.random.randn(d2, 1)
    
W1, W2, b1, b2, loss = neural_network(X_train, Y_train, W1, W2, b1, b2)

# luu gia tri 
W = np.array((W1, W2))
b = np.array((b1, b2))
np.save('Data/W_b/W', W)
np.save('Data/W_b/b', b)

W = np.load('Data/W_b/W.npy', allow_pickle = True)
b = np.load('Data/W_b/b.npy', allow_pickle = True)
W1, W2 = W
b1, b2 = b

from predict import predict

# du doan va so sanh ket qua
y_pred_test = predict(X_test, W1, W2, b1, b2)
y_pred_train = predict(X_train, W1, W2, b1, b2)


print('train: ', accuracy_score(y_train, y_pred_train)*100)
print('test : ', accuracy_score(y_test, y_pred_test)*100)

# hien thi bieu do loss
plt.plot(loss)
plt.show()