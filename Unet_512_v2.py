#%%
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import random
import numpy as np
from data_process import data_visualize
from keras.optimizers import Adam
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

#%%
os.getcwd()

#%%
os.chdir('Data Preprocessing')

#%% initialize data
seed = 42
np.random.seed = seed

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

TRAIN_PATH = 'train/'
TEST_PATH = 'test/'

X1 = np.load(f"{TRAIN_PATH}train_rf1_tiles.npz", allow_pickle=True)
X2 = np.load(f"{TRAIN_PATH}train_rf2_tiles.npz", allow_pickle=True)

X1 = X1.f.XY_tiles
X2 = X2.f.XY_tiles
X_train = np.vstack((X1,X2))
X_train = np.reshape(X_train,(X_train.shape[0],512,512,1))

Y1 = np.load(f"{TRAIN_PATH}train_roi1_tiles.npz", allow_pickle=True)
Y2 = np.load(f"{TRAIN_PATH}train_roi2_tiles.npz", allow_pickle=True)

Y1 = Y1.f.XY_tiles
Y2 = Y2.f.XY_tiles
Y_train = np.vstack((Y1,Y2))
Y_train = np.reshape(Y_train,(Y_train.shape[0],512,512,1))

X_test = np.load(f"{TEST_PATH}test_rf1_tiles.npz", allow_pickle=True)
X_test = X_test.f.XY_tiles
X_test = np.reshape(X_test,(X_test.shape[0],512,512,1))

#%%
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

#%%
image_x = random.randint(0, X_train.shape[0])
data_visualize(np.squeeze(X_train[image_x]), False, False)
data_visualize(np.squeeze(Y_train[image_x]), False, True)

#%%
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
c = [32,64,128,256,512]
# Contraction path
c1 = tf.keras.layers.Conv2D(c[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(c[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(c[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(c[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(c[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(c[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(c[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(c[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(c[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(c[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(c[3], (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(c[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(c[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(c[2], (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(c[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(c[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(c[1], (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(c[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(c[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(c[0], (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(c[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(c[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
#%%
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=tf.keras.optimizers.Adam(
    lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

#%%
# Modelcheckpoint
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
checkpointer = tf.keras.callbacks.ModelCheckpoint('./Unet_512_v2model.h5', verbose=1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs/Unet_512_v2'),
    checkpointer
    ]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=30, callbacks=callbacks)
#%%
1+1
#%%

preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5)
preds_val_t = (preds_val > 0.5)
preds_test_t = (preds_test > 0.5)
#%%
np.savez_compressed("Unet_512/predictions.npz", preds_train = preds_train, preds_val = preds_val, preds_test = preds_test)

#%%
#test
print("hello")


