import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from PreProccessing import LoadData


def divide_dataset(Data, Labels):
    x_train, x_test, y_train, y_test = train_test_split(Data, Labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


print("Initializing...")

data, labels = LoadData.RunProcess()

le = LabelEncoder()
le.fit(labels)
labels_int = le.transform(labels)
labels_one_hot = tf.one_hot(labels_int, depth=10)
labels = np.asarray(labels_one_hot)

X_train, X_test, Y_train, Y_test = divide_dataset(data, labels)


def ShapeX_Data(X_data):
    max_shape = max([item.shape for item in X_data], key=lambda x: np.prod(x))
    padded_data = []
    print("\tpadding data...")
    for item in X_data:
        pad_rows = max_shape[0] - item.shape[0]
        pad_cols = max_shape[1] - item.shape[1]
        padded_item = np.pad(item, [(0, pad_rows), (0, pad_cols)], mode='constant')
        padded_data.append(padded_item)
    print("\tstacking data...")
    X_data = np.stack(padded_data, axis=0)
    print("\tdownsampling...")
    X_data = np.resize(X_data, (16, 16, 10))
    print("\tconverting to tensor...")
    X_data = tf.convert_to_tensor(X_data, dtype=tf.float16)
    print(np.shape(X_data))
    return X_data


def ShapeY_Data(Y_data):
    Y_data = np.asarray(Y_data)
    Y_data = np.stack(Y_data, axis=0)
    Y_data = np.resize(Y_data, (16, 16, 10))
    Y_data = tf.convert_to_tensor(Y_data, dtype=tf.float16)
    return Y_data


print("shaping X_train data:")
X_train = ShapeX_Data(X_train)
print("shaping X_test data:")
X_test = ShapeX_Data(X_test)
print("shaping Y_train data:")
Y_train = ShapeY_Data(Y_train)
print("shaping Y_test data:")
Y_test = ShapeY_Data(Y_test)

print(X_train.shape, Y_train.shape)
print(X_train[1])
print(Y_train[1])

print("creating model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (2, 2), activation='relu', input_shape=(X_train.shape), dtype=tf.float16),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

print("compiling...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("fitting...")
model.fit(X_train, Y_train, epochs=20, batch_size=16)

print("evaluating...")
model.evaluate(X_test, Y_test, verbose=2)
