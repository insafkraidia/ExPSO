from tensorflow import keras
from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from ExPSO import ExPSOClass
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def mlp(particleDimensions, x_train, x_test, y_train, y_test):
    try:
        neurons = int(particleDimensions[0][0])
        epochs = int(particleDimensions[0][1])
        # BUT I NEED TO CONVERT TARGETS INTO BINARY CLASS
        y_train = keras.utils.to_categorical(y_train, numberClasses)
        y_test = keras.utils.to_categorical(y_test, numberClasses)
        # NOW I NEED TO BUILD MLP MODEL
        model = Sequential()
        # FULL CONNECTED LAYER
        model.add(Dense(neurons, input_shape=(numberFeatures,)))
        # ACTIVATION FUNCTION OF FULL CONNECTED LAYER
        model.add(Activation('relu'))
        # LAYER THAT PREVENTS OVERFITTING
        model.add(Dropout(rate=0.1))
        model.add(Dense(50))  # FULL CONNECTED LAYER 2
        # ACTIVATION FUNCTION OF FULL CONNECTED LAYER
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(units=numberClasses))
        model.add(Activation('softmax'))
        # DEFINE PARAMETERS OF MODEL COMPILE
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[
                      'accuracy'])
        # TRAIN MODEL
        model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            verbose=0,  # PROGRESS BAR IS ACTIVE
            batch_size=batch_size,
            validation_split=0.3,
            validation_data=(x_test, y_test)
        )
        # BY DEFAULT BATCH_SIZE IS 32, AND IT'S IMPORTANT TO OVERRIDE THIS
        finalScores = model.evaluate(
            x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        return finalScores[0]
    except:
        raise


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


def ObjFunction(particles):

    try:
        allLosses = mlp(particleDimensions=particles, x_train=x_train, x_test=x_test,
                        y_train=y_train, y_test=y_test)
        return allLosses
    except:
        raise


def main():
    nPop = 30
    runs = 10
    lb = 1
    ub = 500
    D = 2
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt,
                                                       lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()


X, Y, x_train, x_test, y_train, y_test = getDataset(30)

neurons = 100
batch_size = 30
numberFeatures = X.shape[1]
numberClasses = 3
epochs = 10
main()
