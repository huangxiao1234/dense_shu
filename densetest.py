from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
import numpy as np
from PIL import Image

Xtrain = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

#--------------------------将彩色变成灰色--------------------
X_train=[]
for i in range(Xtrain.shape[0]):
    X_train.append(np.array(Image.fromarray(Xtrain[i]).convert('L')))

X_train=np.array(X_train)
batch_size = 32
nb_classes = 62
nb_epoch =5000#5000次能到训练集93准确率，单个字母识别
data_augmentation = True

# input image dimensions
img_rows, img_cols = 22, 15
img_channels = 3

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
#-----------------全连接网络-------------------
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(1024))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=1e-5, decay=0, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)


model.save_weights('shu_captcha_CNN_weights.h5')
json_string = model.to_json()
f=open('shu_captcha_CNN_structure.json','w')
f.write(json_string)
f.close()