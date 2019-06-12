from corpusLoader import CorpusLoader

import os
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

class TextGenerator():
    def __init__(self, cLoader):
        self.cLoader = cLoader
        self.model = None
        self.trainingFolder = self.cLoader.trainingFolder
        self.trainingIn = self.cLoader.shapedX
        self.trainingOut = self.cLoader.shapedY
        self.rCharacters = {}
        self.switchKeyVal(self.cLoader.characters)

    def buildModel(self):
        self.model = Sequential()
        self.model.add(LSTM(320, input_shape=(self.trainingIn.shape[1], self.trainingIn.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.trainingOut.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def switchKeyVal(self, characterMapping):
        for key, value in characterMapping.items():
            self.rCharacters[value] = key

    def trainModel(self, epochs, batchSize):
        self.buildModel()
        progressFile = self.trainingFolder + "/weight-{epoch:02d}-{loss:.4f}.hdf5"
        progress = ModelCheckpoint(progressFile, monitor='loss', verbose=True, save_best_only=True, mode='min')
        progressCallback = [progress]

        self.model.fit(self.trainingIn, self.trainingOut, epochs=epochs, batch_size=batchSize, callbacks=progressCallback)

    def loadModel(self, modelName):
        self.buildModel()
        self.model.load_weights("{0}/{1}".format(self.trainingFolder, modelName))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def loadBestModel(self):
        bestModel = ""
        lowestWeight = 0
        for file in os.listdir(self.trainingFolder):
            if file[-5:] != ".hdf5":
                continue
            weight = float(file[file.rfind("-") + 1:-5])
            if not bestModel or weight < lowestWeight:
                bestModel = file
                lowestWeight = weight
        print("Loaded Model:", bestModel)
        self.loadModel(bestModel)

    def generateText(self, charsToGen):
        seed = numpy.random.randint(0, len(self.cLoader.inputs) - 1)
        pattern = self.cLoader.inputs[seed]
        print('\n\n{0}'.format("".join([self.rCharacters[val] for val in pattern])), end="")
        
        for i in range(charsToGen):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(len(self.rCharacters))
            nextValPrediction = self.model.predict(x, verbose=False)
            index = numpy.argmax(nextValPrediction)
            charPrediction = self.rCharacters[index]
            print(charPrediction, end="")
            pattern.append(index)
            pattern = pattern[1:]
        print("\n")

if __name__ == "__main__":
    cLoader = CorpusLoader("corpora/catinhat.txt", 128)
    cLoader.createTrainingDataset()
    cLoader.shapeData()
    cLoader.printStats()

    textGen = TextGenerator(cLoader)
    textGen.trainModel(epochs=20, batchSize=64)

    print("Model Generated!")

    textGen.loadBestModel()
    textGen.generateText(1000)
