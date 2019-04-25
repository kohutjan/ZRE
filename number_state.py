

class NumberState(object):
    def __init__(self, name, input, token):
        self.name = name
        self.input = input
        self.token = token

    def Next(self):
        pass

    def UpdateToken(self):
        self.token = self.input.token
        self.token.passedNumbers.append(self.name)
