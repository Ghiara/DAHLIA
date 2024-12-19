#!/usr/bin/python

import pickle
from collections import defaultdict
from collections import OrderedDict
import sys


def checkDict(inputData):
    return (type(inputData) == dict or type(inputData) == defaultdict \
            or type(inputData) == OrderedDict)


def checkList(inputData):
    return type(inputData) == list


def recursivePrint(inputData, extraTuple=()):
    if checkDict(inputData):
        for extraInfo, data in inputData.items():
            recursivePrint(data, (extraInfo,) + extraTuple)
    elif checkList(inputData):
        if all([not (checkList(x) or checkDict(x)) for x in inputData]):
            print(" ".join([str(x) for x in extraTuple][::-1]), inputData)
        else:
            for data in inputData:
                recursivePrint(data, extraTuple)
    else:
        print(" ".join([str(x) for x in extraTuple][::-1]), inputData)


def pickleViewer(inputFile):
    inputData = pickle.load(open(inputFile, 'rb'))
    print("Viewing pickle file {0} which contains data of type {1}".format(inputFile,
                                                                           type(inputData)))
    recursivePrint(inputData)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: pickleViewer <input.pkl>"
pickleViewer(sys.argv[1])
