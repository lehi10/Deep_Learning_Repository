from keras.models import Sequential , load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=1,activation='sigmoid'))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen= ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/content/training_set/',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('/content/test_set/',target_size=(64,64),batch_size=32,class_mode='binary')


classifier.compile(loss='mse', optimizer='adam')

classifier=load_model('/content/model.lq') 


#classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=25,validation_data=test_set,validation_steps=2000)

#classifier.save_weights('/content/modelo')
 


import numpy as np
from keras.preprocessing import image

test__image =image.load_img('/content/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test__image =image.load_img('/content/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))

test__image = image.img_to_array(test__image)
test__image = np.expand_dims(test__image,axis=0)
result = classifier.predict(test__image)

prediction =''

training_set.class_indices
if(result[0][0] ==1):
    prediction ='dog'
else :
    prediction = 'cat'
    
print(prediction)






