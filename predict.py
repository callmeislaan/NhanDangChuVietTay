import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    return e_Z / np.sum(e_Z, axis = 0)


def predict(X, W1, W2, b1, b2):
    Z1 = W1.T.dot(X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = W2.T.dot(A1) + b2
    Yhat = softmax(Z2)
    return np.argmax(Yhat, axis = 0)