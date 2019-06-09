from corpusLoader import CorpusLoader
from textGenerator import TextGenerator

if __name__ == "__main__":
    trainingFile = "corpora/studyinscarlet.txt"
    cLoader = CorpusLoader(trainingFile, 100)
    cLoader.createTrainingDataset()
    cLoader.shapeData()
    cLoader.printStats()

    textGen = TextGenerator("training/studyinscarlet", cLoader)
    textGen.trainModel(5, 128)

    print("Model Generated!")

    textGen.loadBestModel()
    textGen.generateText(1000)
