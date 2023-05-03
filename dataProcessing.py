import numpy as np
import pandas as pd
import os
import torch

class DataProcessing():

    def __init__(self, DATAPATH, targetLanguage, device) :

        ''' Create dictionaries and tensor pairs for all three splits. '''
    
        self.device = device # Store device
        
        # Store Special Characters and their integer representation (StartOfWord, EndOfWord, PADding, UNKnown)
        self.SOW = '>'
        self.SOW2Int = 0

        self.EOW = '<'
        self.EOW2Int = 1

        self.PAD = '.'
        self.PAD2Int = 2

        self.UNK = '?'
        self.UNK2Int = 3

        # Load Train, Valid and Test Data
        self.trainPath = os.path.join(DATAPATH, targetLanguage, targetLanguage + "_train.csv")
        self.validPath = os.path.join(DATAPATH, targetLanguage, targetLanguage + "_valid.csv")
        self.testPath = os.path.join(DATAPATH, targetLanguage, targetLanguage + "_test.csv")

        self.train = pd.read_csv(self.trainPath, sep = ",", header = None, names = ["source", "target"])
        self.valid = pd.read_csv(self.validPath, sep = ",", header = None, names = ["source", "target"])
        self.test = pd.read_csv(self.testPath, sep = ",", header = None, names = ["source", "target"])

        self.sourceVocab, self.targetVocab = self.preprocess(self.train["source"].to_list(), self.train["target"].to_list()) # Compute vocabularies

        # Store all four mappings
        self.sourceCharToInt, self.sourceIntToChar = self.sourceVocab
        self.targetCharToInt, self.targetIntToChar = self.targetVocab

        # Create word pairs for all three splits
        self.trainPairs = list(zip(self.train["source"].to_list(), self.train["target"].to_list()))
        self.validPairs = list(zip(self.valid["source"].to_list(), self.valid["target"].to_list()))
        self.testPairs = list(zip(self.test["source"].to_list(), self.test["target"].to_list()))

        # Create tensor pairs for all three splits
        self.trainingTensorPairs = [self.tensorsFromPair(trainPair) for trainPair in self.trainPairs]
        self.validTensorPairs = [self.tensorsFromPair(validPair) for validPair in self.validPairs]
        self.testTensorPairs = [self.tensorsFromPair(testPair) for testPair in self.testPairs]

    def updateConfigs(self, configs) :

        ''' Updates parameters that are not common for all runs. '''

        self.cellType = configs['cellType']
        self.attention = configs['attention']
        self.batchSize = configs['batchSize']
        self.epochs = configs['epochs']
        self.learningRate = configs['learningRate']

    def dictionaryLookup(self, languageCharacters) :
        
        ''' Creates (char -> int) and (int -> char) mapping for characters in a language. '''

        charToInt = dict([(char, i + 4) for i, char in enumerate(languageCharacters)]) # Create an enumeration for the characters and store it in a dictionary
        
        # Add special symbols
        charToInt[self.SOW] = self.SOW2Int
        charToInt[self.EOW] = self.EOW2Int
        charToInt[self.PAD] = self.PAD2Int
        charToInt[self.UNK] = self.UNK2Int
        
        intToChar = dict((i, char) for char, i in charToInt.items()) # Store the reverse mapping of charToInt in intToChar dictionary
        
        return (charToInt, intToChar)

    def preprocess(self, source, target) :
        
        ''' Creates dictionaries. '''

        # Sets of unique characters present in all source language words and all target language words
        sourceChars = set()
        targetChars = set()

        # Converting each element of list to string
        source = [str(x) for x in source]
        target = [str(x) for x in target]

        # Populate SourceChars Set
        for src in source:
            for char in src:
                sourceChars.add(char)

        # Populate TargetChars Set
        for tgt in target:
            for char in tgt:
                targetChars.add(char)

        # Sort the characters in source and target languages
        sourceChars = sorted(list(sourceChars))
        targetChars = sorted(list(targetChars))

        self.numEncoderTokens = len(sourceChars) + 4 # Number of unique characters in source language -> Each character is a token. Source language tokens are encoderTokens # Add 4 for SOW, EOW, PAD, UNK
        self.numDecoderTokens = len(targetChars) + 4 # Number of unique characters in target language -> Each character is a token. Target language tokens are decoderTokens # Add 4 for SOW, EOW, PAD, UNK

        self.numTrainPairs = len(source) # Number of Pairs in Train Data

        # Find maximum length of a word
        self.maxSourceLengthWord = max([len(txt) for txt in source])
        self.maxTargetLengthWord = max([len(txt) for txt in target])

        valSourceMax = max([len(txt) for txt in self.valid["source"].to_list()])
        testSourceMax = max([len(txt) for txt in self.test["source"].to_list()])

        valTargetMax = max([len(txt) for txt in self.valid["target"].to_list()])
        testTargetMax = max([len(txt) for txt in self.test["target"].to_list()])

        self.maxSourceLengthWord = max([self.maxSourceLengthWord, valSourceMax, testSourceMax])
        self.maxTargetLengthWord = max([self.maxTargetLengthWord, valTargetMax, testTargetMax])

        self.maxLengthWord = max(self.maxSourceLengthWord, self.maxTargetLengthWord) 

        # Create the required dictionaries
        sourceVocab = self.dictionaryLookup(sourceChars)
        targetVocab = self.dictionaryLookup(targetChars)

        return sourceVocab, targetVocab

    def tensorFromWord(self, charToInt, word) :

        ''' Returns a tensor of integers of fixed size for a given word. '''
        
        indexes = []
        
        # List of unique integer corresponding to each character
        for char in word :
            if char in charToInt :
                indexes.append(charToInt[char])
            else :
                indexes.append(self.UNK2Int)

        indexes.append(self.EOW2Int) # We represent End of Word with EOW2Int
        indexes.extend([self.PAD2Int] * (self.maxLengthWord - len(indexes) + 1)) # Add Padding after EOW 
        
        return torch.tensor(indexes, dtype = torch.long, device = self.device).view(-1, 1)

    def tensorsFromPair(self, pairOfWords) :
        
        ''' Returns pair of tensors for input pair of words '''

        sourceTensor = self.tensorFromWord(self.sourceCharToInt, pairOfWords[0])
        targetTensor = self.tensorFromWord(self.targetCharToInt, pairOfWords[1])

        return (sourceTensor, targetTensor)
