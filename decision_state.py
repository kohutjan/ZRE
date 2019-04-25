

class DecisionState(object):
    def __init__(self, token):
        self.inputs = []
        self.token = token

    def Next(self):
        # Pick token with maximum value
        self.tmpToken = self.inputs[0].token
        for inputState in self.inputs:
            if inputState.token.value > self.tmpToken.value:
                self.tmpToken = inputState.token

    def UpdateToken(self):
        self.token = self.tmpToken
