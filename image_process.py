import numpy as np

def conv2d(A, W, pad = 0, stride = 1):# tich chap
    n_H_old, n_W_old = A.shape
    f, _ = W.shape
    A_pad = np.pad(A, pad_width = pad, mode = 'constant', constant_values = 0)
    n_H_new = int((n_H_old - f + 2*pad)/stride) + 1
    n_W_new = int((n_W_old - f + 2*pad)/stride) + 1
    RS = np.zeros((n_H_new, n_W_new))
    for h in range(n_H_new):
        for w in range(n_W_new):
            h_start = stride*h
            h_end = h_start + f
            w_start = stride*w
            w_end = w_start + f
            RS[h, w] = int(np.sum(A_pad[h_start:h_end, w_start:w_end]*W))
    return RS

def tim_nguong(I): # theo cong thuc tren mang
    height, width = I.shape
    s = height*width
    h = np.unique(I, return_counts=True)
    g = h[0]
    h = h[1]
    t = np.cumsum(h)
    hg = h*g
    chg = np.cumsum(hg)
    m = (1./t)*chg
    t[-1] = 1
    f = (t/(s - t))*((m-m[-1])**2)
    f[-1] = 0
    i = np.argmax(f)
    return g[i]

def image_process(I):
    I = 255 - I # doi mau de giong voi train data

    # loc trung binh
    W = 1./9*np.ones((3, 3))
    I = conv2d(I, W, pad = 1)
    
    # phan nguong
    nguong = tim_nguong(I)
    RS = np.where(I < nguong, 0, I)

    return RS
