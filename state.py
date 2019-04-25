from math import log

class State(object):
    def __init__(self, name, input, token, likelihoodMatrixColIndex):
        self.name = name
        self.input = input
        self.token = token
        self.likelihoodMatrixColIndex = likelihoodMatrixColIndex
        self.likelihood = 0

    def UpdateLikelihood(self, likelihoodMatrixCol):
        self.likelihood = likelihoodMatrixCol[self.likelihoodMatrixColIndex]

    def Next(self):
        #take max of two incomming tokens (input and self loop)
        if (self.input.token.value + log(0.5)) > (self.token.value + log(0.5)):
            self.tmpToken = self.input.token
        else:
            self.tmpToken = self.token

    def UpdateToken(self):
        self.token = self.tmpToken
        self.token.value += log(0.5) + self.likelihood
