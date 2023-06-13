from tensorflow import keras
from keras.layers import Activation, Dense, Flatten, Conv1D, MaxPooling1D, Activation, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from ExPSO import ExPSOClass
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy


def cnn(x_train, x_test, y_train, y_test, batch_size, epochs, filters, kernel_size, stride=1):
    '''
    THIS 4 PARAMETERS ARE ALREADY NORMALIZED (MIN-MAX NORMALIZATION)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param epochs: integer that represents epochs
    :param filters: integer --> dimensionality of output space(number of output filters in the convolution)
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :return: score of model: accuracy
    '''

    try:

        # I NEED TO RESHAPE DATA TO: (number samples, time step ,features) --> for this example, time_step is 1, and the reshape format is : (samples, features)
        # input shape in channels last --> (time steps, features), if time step is 1, then (None, features) --> https://keras.io/layers/convolutional/
        # TIME STEPS = FEATURES AND FEATURES=TIME STEPS
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # I NEED TO CONVERT TARGETS INTO BINARY CLASS, TO PUT THE TARGETS INTO SAME RANGE OF ACTIVATION OF FUNCTIONS LIKE: SOFTMAX OR SIGMOID
        y_train = keras.utils.to_categorical(y_train, 3)
        y_test = keras.utils.to_categorical(y_test, 3)

        # EXPLANATION BETWEEN PADDING SAME AND VALID: https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        # MODEL CREATION
        input_shape = (x_train.shape[1], 1)
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                  input_shape=input_shape, padding='valid'))  # FIRST CNN
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # I maintain the default value -->  max pool matrix (2.2)
        model.add(MaxPooling1D(strides=1, padding='same'))
        model.add(Flatten())
        model.add(Dense(3))  # FULL CONNECTED LAYER --> OUTPUT LAYER 3 OUTPUTS
        model.add(Activation('softmax'))

        # COMPILE MODEL
        model.compile(optimizer='Adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        # FIT MODEL
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs, verbose=0
        )
        # EVALUATE MODEL
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f'Test loss: {loss:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')

        predict = model.predict(x=x_test, batch_size=batch_size)

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


def getBestNumberOfNodesAndKernelForCNN(X_train, X_test, Y_train, Y_test, params):
    '''
    The objetive of this function is described in objectiveFunctionPSO() function
    Ref: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    :param X_train: array --> samples for train
    :param X_test: array --> samples for test
    :param Y_train: array --> samples for train
    :param Y_test: array --> samples for test
    :param params: represents particle parameters --> 2 parameters --> first one input node value and second one kernel length --> unidimensional array
    :return: error (prevision of samples) of a Particle
    '''

    # TRANSFORM DOUBLE VALUES OF PARTICLE DIMENSION FROM DOUBLE TO INTEGER --> PSO USES DOUBLE VALUES
    # ROUND FUNCTION WAS USED, IN ORDER TO AVOID DOWN UP ROUND'S, IF I DIDN'T CONSIDERED ROUND THIS VALUES ARE DOWN UP (1,6) --> 1, AND I WANT (1,6) --> 2
    params = [int(round(params[i])) for i in range(len(params))]

    # RESHAPE DOS DADOS DE TREINO
    X_train = X_train.reshape(len(X_train), 4, 1)
    X_test = X_test.reshape(len(X_test), 4, 1)

    # CONVERTION OF VECTOR OUTPUT CLASSES TO BINARY
    Y_train = keras.utils.to_categorical(Y_train, 3)
    Y_test = keras.utils.to_categorical(Y_test, 3)

    # EXPLANATION OF INPUT_SHAPE: input_shape needs only the shape of a sample: (timesteps,data_dim)
    # MODEL CREATION --> SEQUENTIAL OPTION, PERMITES TO CREATES A BUILD OF A CNN MODEL
    model = Sequential()
    model.add(Conv1D(params[0], 2, activation='relu', input_shape=(4, 1)))
    # PODIA TER FEITO APENAS MAXPOOLING E TER DEFINIDO UM VALOR PARA A MATRIX, MAS COMO O EXEMPLO É SIMPLES PENSO QUE ASSIM É MELHOR
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    # THREE THE NUMBER OF OUPUTS OF PROBLEM --> FULLY CONNECTED LAYER
    model.add(Dense(3, activation='softmax'))
    model.summary()
    # COMPILE THE MODEL --> 3 ATTRIBUTES https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    # ADAM IS USED TO CONTROL THE RATE LEARNING OF WEIGHTS OF CNN
    # ‘categorical_crossentropy’ for our loss function
    # NAO PRECISAVA DE USAR NADA DISTO --> SERVE APENAS PARA MELHOR ANÁLISE
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, Y_train, epochs=1)

    # RETURNS A NUMPY ARRAY WITH PREDICTIONS
    predictions = model.predict(X_test)

    # WELL, I NEED TO COMPARE THE PREDICTIONS WITH REAL VALUES
    numberRights = 0
    for i in range(len(Y_test)):
        indexMaxValue = numpy.argmax(predictions[i], axis=0)
        # COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
        if indexMaxValue == numpy.argmax(Y_test[i], axis=0):
            numberRights = numberRights + 1

    # HIT PERCENTAGE OF CORRECT PREVISIONS
    hitRate = numberRights / len(Y_test)

    # LOSS FUNCTION --> I VALORIZE PARTICLES IF MINOR VALUES OF NODES AND KERNEL'S BUT I PUT MORE IMPORTANCE IN PARTICLES THAT GIVE MORE ACCURACY RATE
    # I GIVE THIS WEIGHTS IN ORDER TO OBTAIN GOOD SOLUTIONS, WITH LOW VALUE PARAMETERS, THUS REDUCING COMPUTATIONAL POWER
    loss = (1.0 * (1.0 - (1/params[0]))) + (0.5 *
                                            (1.0 - (1/params[1]))) + (2.0 * (1 - hitRate))

    return loss


def ObjFunction(particles):

    try:

        numberFilters = int(particles[0][0])  # FLOAT TO INT
        numberEpochs = int(particles[0][1])
        # CALL CNN FUNCTION cnn --> RETURN accuracy
        accuracy = cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                       epochs=numberEpochs, filters=numberFilters, kernel_size=kernel_size, stride=stride)

        # APPLY LOST FUNCTION --> THE MAIN OBJECTIVE IS TO MINIMIZE LOSS --> MAXIMIZE ACCURACY AND AT SAME TIME MINIMIZE THE NUMBER OF EPOCHS
        # AND FILTERS, TO REDUCE TIME AND COMPUTACIONAL POWER
        loss = 1.5 * ((1.0 - (1.0/numberFilters)) +
                      (1.0 - (1.0/numberEpochs))) + 2.0 * (1.0 - accuracy)
        return loss  # NEED TO RETURN THIS PYSWARMS NEED THIS

    except:
        raise


def main():
    nPop = 30
    runs = 20
    lb = 1
    ub = 500
    D = 2
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt,
                                                       lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()


batch_size = 5
kernel_size = (4,)
stride = 1
X, Y, x_train, x_test, y_train, y_test = getDataset(
    25)  # TEST PERCENTAGE IS 25%
main()
