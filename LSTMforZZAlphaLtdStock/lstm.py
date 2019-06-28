import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import glob
import csv

def month_string_to_number(string):
	m = {
		'Jan': "01",
		'Feb': "02",
		'Mar': "03",
		'Apr':"04",
		 'May':"05",
		 'Jun':"06",
		 'Jul':"07"	,
		 'Aug': "08",
		 'Sep':"09",
		 'Oct':"10",
		 'Nov':"11",
		 'Dec':"12"
		}
	return m[string]


def preprocesarData():
	path = 'dataset/'  
	allFiles=glob.glob(path+"/*/*.txt")
	
	allFiles=sorted(allFiles)

	avgs =[]
	for oneFile in allFiles:
		filepath = oneFile
		with open(filepath) as fp:
			avgPerFile = 0
			count = 0
			line = fp.readline()
			date=""
			while line:
				var= line.split()
				size=len(var)       
				if size==0:
					break
				date = var[2][:4]+"-"+month_string_to_number(var[0])+"-"+var[1]
				avg= float(var[size-1])
				count+=1
				avgPerFile+=avg
		
				line = fp.readline()
			avgPerFile=(avgPerFile/count)
			avgs.append([date,avgPerFile])

	avgs=avgs

	with open('foo.csv', 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerows(avgs)

#Convierte un array de valores en matriz
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


print("Preprocesando DataSet")
preprocesarData()
print("Preprocesamiento Finalizado")

numpy.random.seed(7)

#Cargando dataSet
dataframe = read_csv('foo.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalizar el DataSet
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Obtener data de Entrenamiento y Test
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape  X=t y Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Generar el Modelo

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

#Hacer Predicciones
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calcular RMSE 
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Grafico de entrenamiento
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Grafico de Prueba
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Grafico de predicciones
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

plt.show()
