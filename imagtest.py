from PIL import Image
from scipy.misc import imread
from numpy import *
# img = imread('1.jpg')
# print(img.shape)
# im=Image.open('1.jpg')
# im=array(im)
# k=Image.fromarray(im)
# k.save('3.jpg')
X_train = load("X_train.npy")
Y_train = load("Y_train.npy")
Xtrain=[]
for i in range(X_train.shape[0]):
    Xtrain.append(array(Image.fromarray(X_train[i]).convert('L')))
print(shape(array(Xtrain)))