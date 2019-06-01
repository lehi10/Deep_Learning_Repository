import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import keras


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def build(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units =6,kernel_initializer = 'uniform', activation='relu',input_dim=11))
    
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build)
parameters = {
    'batch_size':[50,100],
    'epochs'  :[100,500],
    'optimizer' : ['adam','rmsprop']
}

grid_search = GridSearchCV(estimator = classifier,
    param_grid = parameters,
    scoring='accuracy',
    cv = 10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)
