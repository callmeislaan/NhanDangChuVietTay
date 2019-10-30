from tkinter import Tk
from tkinter import Button
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from predict import predict
from tkinter import messagebox
import matplotlib.pyplot as plt
from image_process import image_process
from os import path

# load Weight and bias
W1, W2 = np.load('Data/W_b/W.npy', allow_pickle = True)
b1, b2 = np.load('Data/W_b/b.npy', allow_pickle = True)

filename = ''

# for load file and display images
def load_file():
    global filename
    img_path = path.join('Data/Images')
    filename = filedialog.askopenfilename(initialdir=img_path, title='Select a file', filetypes=(('Image', '*.png *.jpg *.jpeg'), ('All files', '*.*')))
    img = Image.open(filename)
    image = ImageTk.PhotoImage(img.resize((400, 350)))
    image_gray = ImageTk.PhotoImage(img.convert('L').resize((400, 350)))
    label = Label(root, image = image)
    label.image = image
    label.grid(row = 1, column = 0)
    label_gray = Label(root, image = image_gray)
    label_gray.image = image_gray
    label_gray.grid(row = 1, column = 1)

def predict_image():
    if filename == '':
        messagebox.showerror(title='Lỗi chưa chọn ảnh', message='Chưa có ảnh để dự đoán, vui lòng chọn ảnh!')
    I = Image.open(filename).convert('L').resize((28, 28))
    I = np.array(I)
    # xu ly anh
    I = image_process(I)
    # du doan
    rs = predict(I.reshape(28*28, 1), W1, W2, b1, b2)

    if rs == 0:
        r = 'A'
    elif rs == 1:
        r = 'B'
    elif rs == 2:
        r = 'C'
    else:
        r = 'D'

    messagebox.showinfo(title='Kết quả dự đoán', message='Chữ trong ảnh là chữ: ' + r)

# build GUI
root = Tk()
root.minsize(800, 400)
root.title('Nhận dạng chữ viết tay')

btn_load = Button(root, text = 'Tải ảnh', command = load_file)
btn_load.grid(row = 0, column = 0)
btn_predict =Button(root, text = 'Dự đoán', command = predict_image)
btn_predict.grid(row = 0, column = 1)

root.mainloop()