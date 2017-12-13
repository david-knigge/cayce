import sys, requests, json, datetime, os
clear = lambda: os.system('cls')
clear()
print("Welcome to Cayce v0.01 a")
print("  ")
print("Loading dependencies. Please wait.")

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import numpy as np
#from pyHook import HookManager
#from pyHook.HookManager import HookConstants
#from pprint import pprint

#warnings.filterwarnings("ignore")

day = 86400
now = int(time.time())

def main():

    # Retrieve input from user
    clear = lambda: os.system('cls')
    clear()

    print("Welcome to Cayce v0.01 a")
    print("  ")

    print("Enter currency (Ex. BTC)")
    currency = input("")

    while(not checkInput(currency)):
        printhead()
        print("Currency not found, please try again. For a full list of available currencies press ctrl + h")
        print("Enter currency (Ex. BTC)")
        currency = input("")

    data_raw = retrievedata(currency)
    sample_size = len(data_raw)
    data = [unit[2] for unit in data_raw]

    sequence_length = 50 + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = normalise_windows(result)

    result = np.array(result)

    print(result.shape)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    printhead()
    print("Creating Model.")
    model =  createmodel()
    print("Sample size: ",sample_size)
    model.fit(x_train, y_train, batch_size=512, epochs=1, validation_split=0.05)
    print(x_train)
    print("Input what the fuck yes hhello")

def printhead():
    clear = lambda: os.system('cls')
    clear()

    print("Welcome to Cayce v0.01 a")
    print("  ")


def checkInput(currency):

    r = requests.get('https://min-api.cryptocompare.com/data/pricehistorical?tsyms=USD&ts=' + str(now) + '&fsym=' + currency)

    if(r.status_code == requests.codes.ok and 'Response' not in r.json()):
        return True
    else:
        return False

def retrievedata(currency):

    training_data = []
    timestamp = now

    clear = lambda: os.system('cls')
    clear()

    r = requests.get('https://min-api.cryptocompare.com/data/pricehistorical?tsyms=USD&ts=' + str(timestamp) + '&fsym=' + currency)

    i = 0
    previous = r.json()[currency]['USD']

    while(r.status_code == requests.codes.ok and i < 500):

        clear()
        print("Welcome to Cayce v0.01 a")
        print("  ")
        print("Retrieving training data for " + currency + " on "+ time.ctime(timestamp))

        percentage = (r.json()[currency]['USD'] - previous) / previous

        training_data.append([timestamp,i,percentage])

        i = i + 1
        timestamp -= day

        r = requests.get('https://min-api.cryptocompare.com/data/pricehistorical?tsyms=USD&ts=' + str(timestamp) + '&fsym=' + currency)

    training_data.reverse()

    for unit in training_data:
        unit[1] = len(training_data) - unit[1]

    return training_data

def createmodel():
    model = Sequential()
    model.add(LSTM(return_sequences=True, input_shape=(None, 1), units=50))
    model.add(Dropout(0.2))

    model.add(LSTM(100,return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('Done. Compilation time : ', time.time() - start)
    return model

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

'''
def listcurrencies():
    r = request.get('https://www.cryptocompare.com/api/data/coinlist/')
    response = r.json()

    pprint(currency for currency in response)


def OnKeyboardEvent(event):
    ctrl_pressed = HookManager.GetKeyState(HookConstants.VKeyToID('VK_CONTROL') >> 15)
    if ctrl_pressed and HookConstant.IDToName(event.keyId) == 'h':
        listcurrencies()
'''

main()
