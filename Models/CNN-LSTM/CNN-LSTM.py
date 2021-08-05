import numpy as np
import matplotlib.pyplot as plt
import random , pickle , cv2 ,os,datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Activation,TimeDistributed
from tensorflow.keras.layers import Conv2D , LSTM ,Input ,MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K

dense_layers =[1,2,3]
layer_sizes =[64,128]
lstm_layers = [2,3]

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA = "E:\DATA"

CATEGORIES = ["Fire","Forrest"]

for category in CATEGORIES : #
    path = os.path.join(DATA,category) #yangın ve orman için tüm imgleri patch içine alır
    for img in os.listdir(path): #her img array haline çevirme
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) #grileştirme de yapıldı.
        plt.imshow(img_array,cmap = 'gray') # grafiklertieme
        #plt.show()

        break
    break

#print(img_array.shape)#kaç satır kaç sütüun yazdırma.
IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
#plt.show()
# üstte datalarımızı array ve gray uygun hale getirdik.şimdi data eğitimi yapcağız.

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA,category)
        class_num = CATEGORIES.index(category) #sınıflandırma yapıyoruz. 0=yangın,1=orman(yangınsız)
        for img in os.listdir(path):  # her img array haline çevirme
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # grileştirme de yapıldı.
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

#print(len(training_data))

random.shuffle(training_data)
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25,random_state=42)
X_train = tf.reshape(X_train,(429,-1,IMG_SIZE,IMG_SIZE,1))
X_test = tf.reshape(X_test,(143,-1,IMG_SIZE,IMG_SIZE,1))

print(X_train.shape)
print(X_test.shape)
def precision_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1_score(y_true,y_pred):
    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for lstm_layer in lstm_layers:
            NAME = "{}-CNN-LSTM kat sayısı -{}-hücre sayısı-{}-dense katman sayısı-{}".format(lstm_layer, layer_size, dense_layer, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            print(NAME)

            cnn = Sequential()

            cnn.add(TimeDistributed(Conv2D(32, (5, 5), input_shape=(100,100,1))))
            cnn.add(TimeDistributed(Activation('relu')))
            cnn.add(TimeDistributed(MaxPool2D(pool_size=(4,4))))
            #cnn.add(TimeDistributed(Flatten()))

        for l in range(lstm_layer-1):
            lstm = Sequential()
            lstm.add(LSTM(layer_size, return_sequences=True))
            lstm.add(Activation('relu'))


        #model.add(Flatten())  # 3 vektörü 1d çevirme fonksiyonu
        for l in dense_layers:
            dense = Sequential()
            dense.add(Dense(layer_size) )
            dense.add(Activation('relu'))



            dense.add(Dense(1))
            dense.add(Activation('softmax'))
            tboard_log_dir = os.path.join("../Models/CNN-LSTM/4760logs/CNN-GRU/25-01-21"
                                          , NAME)
            tensorboard = TensorBoard(log_dir=tboard_log_dir)

            main_input = Input(shape=(10,IMG_SIZE,IMG_SIZE,1))

            model = cnn(main_input)
            model = GRU(model)
            model = dense(model)
            print('lstm:',lstm.output_shape)
            print('dense:' ,dense.output_shape)
            final_model = Model(inputs=main_input,outputs = model)

            final_model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy',
                                   f1_score,
                                   recall_score,
                                   precision_score])

            final_model.fit(X_train, y_train,
                      batch_size=8,
                      epochs=20,
                      validation_split=0.3,
                      callbacks=[tensorboard])
