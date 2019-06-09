import ast
import os
import numpy
from tensorflow.keras import utils

class CorpusLoader:
    def __init__(self, corpusLoc=None, inputLength=50):
        self.corpusLoc = corpusLoc
        self.rawText = ""
        self.characters = set()
        self.uniqueCharacters = 0
        self.inputLength = inputLength
        self.inputs = []
        self.outputs = []
        self.shapedX = []
        self.shapedY = []
        if corpusLoc is not None:
            self.readCorpus(corpusLoc)

    def readCorpus(self, corpusLoc):
        with open(corpusLoc, "r") as corpusReader:
            self.rawText = corpusReader.read()
            characterSet = sorted(list(set(self.rawText)))
            self.characters = dict((character, index) for index, character in enumerate(characterSet))

    def createTrainingDataset(self):
        if not self.rawText:
            return

        for i in range(len(self.rawText) - self.inputLength):
            trainingX = self.rawText[i:i + self.inputLength]
            trainingX = [self.characters[character] for character in trainingX]
            trainingY = self.characters[self.rawText[i + self.inputLength]]
            self.inputs.append(trainingX)
            self.outputs.append(trainingY)

    def shapeData(self):
        if not self.uniqueCharacters:
            self.uniqueCharacters = len(self.characters)
        self.shapedX = numpy.reshape(self.inputs, (len(self.inputs), self.inputLength, 1)) / float(self.uniqueCharacters)
        self.shapedY = utils.to_categorical(self.outputs)

    def writeDataset(self, datasetName=None):
        if datasetName is None:
            datasetName = self.corpusLoc[self.corpusLoc.rfind("/") + 1:self.corpusLoc.rfind(".")]
        trainingFolder = "training/{}".format(datasetName)
        if not os.path.isdir(trainingFolder):
            os.mkdir(trainingFolder)
        with open("training/{}/train.txt".format(datasetName), "w") as trainingWriter:
            trainingWriter.write("{0},{1}".format(self.inputLength, len(self.characters)) + "\n")
            for i in range(len(self.inputs)):
                trainingWriter.write("{0}:{1}\n".format(self.inputs[i], self.outputs[i]))

    def loadDataset(self, datasetName):
        with open("training/{}/train.txt".format(datasetName), "r") as trainingReader:
            firstLine = True
            for line in trainingReader.readlines():
                if firstLine:
                    header = line.split(",")
                    self.inputLength = int(header[0])
                    self.uniqueCharacters = int(header[1])
                    firstLine = False
                    continue
                inOut = line.split(":")
                self.inputs.append(ast.literal_eval(inOut[0]))
                self.outputs.append(int(inOut[1]))
        print("Dataset Loaded")

    def printStats(self):
        print("Input Length:", self.inputLength)
        print("Corpus Character Count:", len(self.rawText))
        print("Unique Character Count:", len(self.characters))
        print("Patterns:", len(self.inputs))
        print("Unique Patterns:", len(set(tuple(i) for i in self.inputs)))

if __name__ == "__main__":
    # cLoader = CorpusLoader("corpora/studyinscarlet.txt", 50)
    # cLoader.createTrainingDataset()
    # cLoader.printStats()
    # cLoader.writeDataset()
    cLoader = CorpusLoader()
    cLoader.loadDataset("studyinscarlet")
    cLoader.shapeData()
