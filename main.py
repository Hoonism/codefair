# import amp as amp
import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from keras import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation, Flatten
from sklearn.model_selection import train_test_split

mat = scipy.io.loadmat('wiki_crop/wiki.mat')
instances = mat['wiki'][0][0][0].shape[1]

columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

df = pd.DataFrame(index=range(0, instances), columns=columns)

for i in mat:
    if i == "wiki":
        current_array = mat[i][0][0]
for j in range(len(current_array)):
    df[columns[j]] = pd.DataFrame(current_array[j][0])

def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    exact_date = datetime.fromordinal(int(datenum)) \
                 + timedelta(days=int(days)) + timedelta(hours=int(hours)) \
                 + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) \
                 - timedelta(days=366)

    return exact_date.year
df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)
df['age'] = df['photo_taken'] - df['date_of_birth']

df = df[df['face_score'] != -np.inf]
df = df[df['second_face_score'].isna()]
df = df[~df['gender'].isna()]
df = df[df['face_score'] >= 3]
df = df.drop(columns=['name', 'face_score', 'second_face_score', 'date_of_birth', 'face_location'])
df = df[df['age'] <= 100]
df = df[df['age'] >= 0]


target_size = (166, 166)

def getImagePixels(image_path):
    img = tf.keras.preprocessing.image.load_img("wiki_crop/%s" % image_path[0], grayscale=False, target_size=target_size)
    x = tf.keras.preprocessing.image.img_to_array(img).reshape(1, -1)[0]
    return x

df['pixels'] = df['full_path'].apply(getImagePixels)
print(df.head())

classes = 101
target = df['age'].values
target_classes = tf.keras.utils.to_categorical(target, classes)

features = []

for i in range(0, df.shape[0]):
    features.append(df['pixels'].values[i])

features = np.array(features)
features = features.reshape(features.shape[0], 166, 166, 3)
train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.20)

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(166, 166, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

print(model.summary)
model.load_weights('vgg_face_weights.h5')

for layer in model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

age_model = Model(inputs=model.input, outputs=base_model_output)

age_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='age_model.hdf5', monitor="val_loss", verbose=1, save_best_only=True, mode='auto')

scores = []
epochs = 100
batch_size = 64
for i in range(epochs):
    print("epoch ", i)
    ix_train = np.random.choice(train_x.shape[0], size=batch_size)
    score = age_model.fit(train_x[ix_train], train_y[ix_train], epochs=1, validation_data=(test_x, test_y), callbacks=[checkpointer])
    scores.append(score)