class CorpusLoader:
    def __init__(self, corpusLoc):
        self.corpusLoc = corpusLoc
        self.rawText = ""
        self.characters = set()
        self.inputs = []
        self.outputs = []
        self.readCorpus(corpusLoc)

    def readCorpus(self, corpusLoc):
        with open(corpusLoc, "r") as corpusReader:
            self.rawText = corpusReader.read()
            self.characters = sorted(list(set(self.rawText)))
            print(self.characters)

if __name__ == "__main__":
    cLoader = CorpusLoader("corpora/sherlockholmes.txt")
