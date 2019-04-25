import numpy as np
import os
import sys
import argparse
sys.path.insert(0, 'tools')
from htk import readhtk
from collections import OrderedDict

from state import State
from token import Token

from number_model import NumberModel
from net_model import NetModel
from number_state import NumberState
from decision_state import DecisionState

megaNegNumber = -1000000000000

def parseargs():
    print( ' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('--likelihood-matrix', type=str, required=True, help="Input likelihood matrix to recognize.")
    parser.add_argument('--phonemes', type=str, required=True, help="phonemes file.")
    parser.add_argument('--zre-dict', type=str, required=True, help="zre.dict file.")
    args = parser.parse_args()
    return args


def main():
    global megaNegNumber

    args = parseargs()
    likelihoodMatrix = readhtk(args.likelihood_matrix)
    print(likelihoodMatrix)
    print(likelihoodMatrix.shape)
    print(likelihoodMatrix[0].shape)

    phonemesMapping = GetPhonemesMapping(args.phonemes)
    print("PHONEMES MAPPING")
    for phonem, index in phonemesMapping.items():
        print(phonem, index)
    print("")

    numberModels = GetNumberModels(args.zre_dict, phonemesMapping)
    print("NUMBER MODELS")
    for numberModel in numberModels:
        print(numberModel.name)
        for state in numberModel.states:
            print(state.name, state.token.value)
        print("")
    print("")

    netModel = NetModel(numberModels, DecisionState(Token(megaNegNumber)))


    """
    MAIN LOOP
    """
    for t in range(likelihoodMatrix.shape[0]):
        netModel.UpdateModelsLikelihoods(likelihoodMatrix[t])
        netModel.Next()
        netModel.UpdateTokens()
        """
        print(likelihoodMatrix[t])
        for numberModel in netModel.numberModels:
            for state in numberModel.states:
                print(state.name, state.likelihoodMatrixColIndex, state.value)
            break
        break
        """


def GetNumberModels(zreDict, phonemesMapping):
    numberModels = []
    with open(zreDict) as f:
        hmmModelsSpecificationLines = f.readlines()
        for hmmModelSpecification in hmmModelsSpecificationLines:
            numberModel = NumberModel(hmmModelSpecification.strip().split()[0])
            hmmModelSpecification = hmmModelSpecification.strip().split()[1:]
            print(numberModel.name)
            previousState = None
            for phonem in hmmModelSpecification:
                # Generate three sub states fo each phonem
                for subPhonemIndex in range(3):
                    stateName = "{}_{}".format(phonem, subPhonemIndex)
                    likelihoodMatrixColIndex = phonemesMapping[phonem] * 3 + subPhonemIndex
                    previousState = State(stateName, previousState, Token(megaNegNumber), likelihoodMatrixColIndex)
                    numberModel.AddState(previousState)
            # Add dummy number state
            numberModel.AddNumberState(NumberState(numberModel.name, previousState, Token(megaNegNumber)))
            numberModels.append(numberModel)
    return numberModels


def GetPhonemesMapping(phonemesFile):
    phonemesMapping = OrderedDict()
    with open(phonemesFile) as f:
        for i, line in enumerate(f.readlines()):
            phonemesMapping[line.strip()] = i
    return phonemesMapping


if __name__ == '__main__':
    main()
