from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import time
from matplotlib import pyplot as plt

# Setting the start variable for time measurement
start_time = time.time()

# Variable
max_features = 5000
maxlen = 80  # 단어중에 가장 많이 쓰이는 단어중 가장 일반적인 단어 80개로 자르고 나머지는 0으로 ..
batch_size = 32

print('Loading the data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train data sequences')
print(len(x_test), 'test data sequences')

print(x_train[0])
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

#
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(y_train.shape)
print(y_train)


#
# print('Build model...')
# model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# print('Train...')
# hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=5, batch_size=batch_size, verbose=1)
# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
# print('Test score:', score)
# print('Test accuracy:', acc)
#
# # model visualize
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()
#
# # using svg visual model
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#
# # spent to time
# print("--- %s seconds ---" % (time.time() - start_time))