import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import to_categorical
#from keras.utils import np_utils
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
print(f"Shape of input data -> {train_images.shape}")

fig = plt.figure(figsize=(13,13))
fig.tight_layout(pad=0.8, h_pad=2)
for idx in range(25):
    ax = fig.add_subplot(5, 5, idx+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    ax.imshow(train_images[idx], cmap=plt.cm.binary)
plt.show()

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)) )
model.add(Conv2D(32, (3, 3), padding='same', activation='relu') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu') )
model.add(Conv2D(16, (3, 3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, (3, 3), padding='same') )
model.add(Conv2D(8, (3, 3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax')) # Dense(10) because we have 10classes in this example
model.summary()

train_images = train_images/255.0
test_images = test_images/255.0
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=[test_images, test_labels])

plt.plot(history.history['accuracy'], label='training accuracy')
plt.legend()
plt.show()

accuracy_loss = model.evaluate(test_images, test_labels, verbose=2)
print('Test Loss', accuracy_loss[0])
print('Test Accuracy', accuracy_loss[1])

#print(model.predict([test_images[0]]))
single_image = test_images[0]
plt.imshow(single_image.squeeze(), cmap=plt.cm.binary)
plt.title("Actual Label: {}".format(np.argmax(test_labels[0])))
plt.show()

# Reshape and predict
single_image = np.expand_dims(single_image, axis=0)
prediction = model.predict(single_image)
predicted_class = np.argmax(prediction)

print("Predicted Class:", predicted_class)
print("Confidence Scores:", prediction)