import sys
sys.path.insert(0, 'tools')
from token import Token

class DecisionState(object):
    def __init__(self, token):
        self.inputs = []
        self.token = token

    def GetTokenCopy(self):
        t = Token(self.token.value)
        for n in self.token.passedNumbers:
            t.AddPassedNumber(n)
        return t

    def Next(self):
        # Pick token with maximum value
        self.tmpToken = self.inputs[0].token
        for inputState in self.inputs:
            if inputState.token.value >= self.tmpToken.value:
                self.tmpToken = inputState.GetTokenCopy()

    def UpdateToken(self):
        self.token = self.tmpToken
