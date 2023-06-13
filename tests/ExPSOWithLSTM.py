from tensorflow import keras
from keras.layers import Activation, LSTM, Dense, Dropout
from keras.models import Sequential
import keras
from ExPSO import ExPSOClass
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy


# LSTM
def lstm(x_train, x_test, y_train, y_test, neurons, epochs):
    '''
    THIS 4 PARAMETERS ARE ALREADY NORMALIZED (MIN-MAX NORMALIZATION)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param epochs: integer that represents epochs
    :param neurons: number of neurons on lstm layer
    :return: score of model: accuracy
    '''

    try:

        # FIRST, I NEED TO RESHAPE DATA (samples, timesteps, features) --> in this example timesteps = features and
        # SWAP BETWEEN TIME STEPS AND FEATURES, LIKE HAPPENS IN CNN
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # I NEED TO CONVERT TARGETS INTO BINARY CLASS, TO PUT THE TARGETS INTO SAME RANGE OF ACTIVATION OF FUNCTIONS LIKE: SOFTMAX OR SIGMOID
        y_train = keras.utils.to_categorical(y_train, 3)
        y_test = keras.utils.to_categorical(y_test, 3)

        # MODEL CREATION
        model = Sequential()
        # batch_input_shape = (batch_size, x_train.shape[1], 1)
        input_shape = (x_train.shape[1], 1)
        # STATEFUL = FALSE, BECAUSE NO DEPENDENCY BETWEEN DATA, AND RETURN SEQUENCES = FALSE, BECAUSE NOW I DON'T NEED TO CREATE A STACKED LSTM
        model.add(LSTM(neurons, input_shape=input_shape,
                  stateful=False, return_sequences=False))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        # model.add(Flatten())
        model.add(Dense(3))
        model.add(Activation('softmax'))

        # COMPILE MODEL
        model.compile(optimizer='Adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        # FIT MODEL
        model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            verbose=0,
            shuffle=True  # IF I USE STATEFUL MODE, THIS PARAMETER NEEDS TO BE EQUALS TO FALSE
        )

        # EVALUATE MODEL
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        print(f'Test loss: {loss:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')

        predict = model.predict(x=x_test)

        predict = (predict == predict.max(axis=1)[:, None]).astype(int)

        numberRights = 0
        for i in range(len(y_test)):
            indexMaxValue = numpy.argmax(predict[i], axis=0)
            if indexMaxValue == numpy.argmax(y_test[i],
                                             axis=0):  # COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
                numberRights = numberRights + 1

        # HIT PERCENTAGE OF CORRECT PREVISIONS
        hitRate = numberRights / len(y_test)

        return hitRate

    except:
        raise

# LSTMPSO


def getDataset(testSize):
    '''
    :param testSize:  integer value between 0-100
    :return: six dataset's --> original input dataset, original output dataset, train dataset(input and output) and test dataset(input and output)
    '''

    trainPercentage = testSize / 100
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # APPLY NORMALIZATION TO ATTRIBUTES --> EXPLANATION OF THIS APPROACH ON FUNCION
    X = applyNormalization(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=trainPercentage)

    return X, Y, x_train, x_test, y_train, y_test


def applyNormalization(X):
    '''
    I make a pre analysis and i conclude that the diferent attributes have distant scales
    and in order to optimize the neural network learning, i have decided to apply the min-
    max technique, a normalization technique
    :param X: data of dataset
    :return: X normalized
    '''

    scaler = MinMaxScaler()

    # FIT AND TRANSFORM - X
    scaler.fit(X)
    X = scaler.transform(X)

    return X


def ObjFunction(particle):

    try:
        neurons = int(particle[0][0])
        epochs = int(particle[0][1])
        # CALL LSTM_MODEL function
        accuracy = lstm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                        neurons=neurons, epochs=epochs)
        # APPLY COST FUNCTION --> THIS FUNCTION IS EQUALS TO CNN COST FUNCTION
        loss = 1.5 * ((1.0 - (1.0/neurons)) + (1.0 - (1.0/epochs))
                      ) + 2.0 * (1.0 - accuracy)
        return loss

    except:
        raise


def main():
    nPop = 30
    runs = 20
    lb = 1
    ub = 200
    D = 2
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt,
                                                       lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()


X, Y, x_train, x_test, y_train, y_test = getDataset(25)
batch_size = 32
main()
