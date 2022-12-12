import glob
import os
import threading

import numpy as np
import tensorflow as tf
import wavfile

base_path = 'C:\dev\CI642\Music_Genre_Classification\Data\genres_original'
data = []
labels = []
print("Initializing...")


def process_file(file_path):
    try:
        audio_data, sr, br = wavfile.read(file_path)
        audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
        data.append(audio_data)
        labels.append(item)
        print(f"Processed file: {file_path}")
    except:
        print(f"Unknown error occurred while processing file: {file_path}\nSkipping...")


threads = []

for item in os.listdir(base_path):
    path = os.path.join(base_path, item)
    print(f"Working on folder: {path}")
    for file_path in glob.glob(os.path.join(path, '*.wav')):
        thread = threading.Thread(target=process_file, args=(file_path,))
        threads.append(thread)
        thread.start()

for thread in threads:
    thread.join()

print("Finished processing all files")

data = np.asarray(data)
labels = np.asarray(labels)


def divide_dataset(x, y, test_size=0.2):
    np.random.seed(0)
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]
    num_test_samples = int(len(x) * test_size)
    x_test = x[:num_test_samples]
    y_test = y[:num_test_samples]
    x_train = x[num_test_samples:]
    y_train = y[num_test_samples:]
    return x_test, x_train, y_test, y_train


model = tf.keras.Sequential([
    tf.keras.layers.Reshape((6990, 130, 13)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

X_test, X_train, Y_test, Y_train = divide_dataset(data, labels)

model.fit(X_train, Y_train, epochs=10)

model.evaluate(X_test, Y_test, verbose=2)
