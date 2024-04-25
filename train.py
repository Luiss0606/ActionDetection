import os
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from config import GlobalVars


label_map = {label:num for num, label in enumerate(os.listdir(GlobalVars.DATA_PATH))}

sequences, labels = [], []
for action in GlobalVars.ACTIONS:
    for sequence in np.array(os.listdir(os.path.join(GlobalVars.DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(GlobalVars.SEQUENCE_LENGTH):
            res = np.load(os.path.join(GlobalVars.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(GlobalVars.ACTIONS), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

# Save the model


last_model = 'model_{}.h5'.format(int(time.time()))
model.save(last_model)

# Save the name of the model in a text file


with open('last_model.txt', 'w') as f:
    f.write(last_model)

if input('Show evaluation? (y/n)') == 'y':
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(multilabel_confusion_matrix(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))