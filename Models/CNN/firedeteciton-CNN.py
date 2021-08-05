import numpy as np
import matplotlib.pyplot as plt
import random , pickle , cv2 ,os,datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential ,load_model,save_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn import metrics

dense_layers =[1,2]
layers_sizes =[32,64]
conv_layers =[2,3]

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

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
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=42)



for dense_layer in dense_layers:
    for layer_size in layers_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())  # 3 vektörü 1d çevirme fonksiyonu
            for l in dense_layers:
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            #tensorboard log dosyalrını kaydetme
            tboard_log_dir = os.path.join("4820logs/26-04-21", NAME)
            tensorboard = TensorBoard(log_dir=tboard_log_dir)
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy',
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.TrueNegatives(),
                                   tf.keras.metrics.TruePositives()
                                   ])

            model.fit(X_train, y_train,
                      batch_size=8,
                      epochs=15,
                      validation_split=0.3,
                      callbacks=[tensorboard])
            save_model(model,'E:/PycharmProjects/ML/Final-Project/CNN/my_cnn.tflearn')
            mymod=load_model('E:/PycharmProjects/ML/Final-Project/CNN/my_cnn.tflearn')
            #test_img = cv2.resize(cv2.imread('E:/PycharmProjects/ML/Dataset/test_image/forest.JPG',cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            #test_img = np.array(test_img).reshape( -1,IMG_SIZE, IMG_SIZE,1)
            #prediction = mymod.predict({'input': test_img})
            #test_image=np.array(test_img).reshape( IMG_SIZE, IMG_SIZE)
            #plt.imshow(test_image,cmap='gray')
            #plt.show()

            #eğtilen örneklerden 32 örnek alınıp, (epoch) 7 adımda eğitim , test train oranı 70'e 30
