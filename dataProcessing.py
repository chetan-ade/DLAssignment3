import numpy as np
import pandas as pd
import os
import torch

class DataProcessing():

    def __init__(self, DATAPATH, targetLanguage, device) :

        ''' Data PreProcessor to compute Encoder Decoder Input. 

            INPUT :     DATAPATH contains folder path where all language directories are present.
                        targetLanguage specifies the folder where the three split csv files are present. 
                        device on which tensors are stored
                
            OUTPUT :    Computes 4 Dictionaries :
                            Source Character To Integer
                            Target Character To Integer 
                            Source Integer to Character
                            Target Integer to Character '''
    
        # Store device
        self.device = device

        # Start Of Word Token
        self.SOW = '>'
        self.SOW2Int = 0

        # End of Word Token
        self.EOW = '<'
        self.EOW2Int = 1

        # Padding Token
        self.PAD = '.'
        self.PAD2Int = 2
        
        # Unknown Token
        self.UNK = '?'
        self.UNK2Int = 3

        # File Path for Train CSV File
        self.trainPath = os.path.join(DATAPATH, targetLanguage, targetLanguage + "_train.csv")

        # File Path for Validation CSV File
        self.validPath = os.path.join(DATAPATH, targetLanguage, targetLanguage + "_valid.csv")

        # File Path for Test CSV File
        self.testPath = os.path.join(DATAPATH, targetLanguage, targetLanguage + "_test.csv")

        # Load the Train Data
        self.train = pd.read_csv(self.trainPath, sep = ",", header = None, names = ["source", "target"])

        # Load the Validation Data
        self.valid = pd.read_csv(self.validPath, sep = ",", header = None, names = ["source", "target"])

        # Load the Test Data
        self.test = pd.read_csv(self.testPath, sep = ",", header = None, names = ["source", "target"])

        # Compute sourceVocab and targetVocab
        self.sourceVocab, self.targetVocab = self.preprocess(self.train["source"].to_list(), self.train["target"].to_list())

        # Once the dictionaries have been populated, store use them for further use
        self.sourceCharToInt, self.sourceIntToChar = self.sourceVocab
        self.targetCharToInt, self.targetIntToChar = self.targetVocab

    def dictionaryLookup(self, languageCharacters) :
        
        ''' Input ->    languageCharacters : Sorted List of characters present in a language

            Output ->   charToInt : Dictionary that maps each character to an Integer including the SOW, EOW, PAD, UNK.
                        intToChar : Dictionary that stores the reverse mapping of charToInt.'''

        # Create an enumeration for the characters and store it in a dictionary
        charToInt = dict([(char, i + 4) for i, char in enumerate(languageCharacters)])

        # Add SOW Token
        charToInt[self.SOW] = self.SOW2Int

        # Add EOW Token
        charToInt[self.EOW] = self.EOW2Int
        
        # Add PAD Token
        charToInt[self.PAD] = self.PAD2Int

        # Add UNK Token
        charToInt[self.UNK] = self.UNK2Int
        
        # Store the reverse mapping of charToInt in intToChar dictionary
        intToChar = dict((i, char) for char, i in charToInt.items())

        # Return the two dictionaries
        return (charToInt, intToChar)

    def preprocess(self, source, target) :
        
        ''' Input ->    Source : Source Column of DataSet converted to a list.
                        Target : Target Column of DataSet converted to a list.    

            Output ->   sourceVocab : Tuple of source language dictionaries.
                        targetVocab : Tuple of target language dictionaries. '''

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

        # Number of unique characters in source language -> Each character is a token. Source language tokens are encoderTokens
        # Add 4 for SOW, EOW, PAD, UNK
        self.numEncoderTokens = len(sourceChars) + 4

        # Number of unique characters in target language -> Each character is a token. Target language tokens are decoderTokens
        # Add 4 for SOW, EOW, PAD, UNK
        self.numDecoderTokens = len(targetChars) + 4

        # Maximum length of a source word
        self.maxSourceLength = max([len(txt) for txt in source])

        # Maximum length of a target word
        self.maxTargetLength = max([len(txt) for txt in target])

        # Number of Pairs in Train Data
        self.numTrainPairs = len(source)

        print("Number of Source-Target Pairs :", self.numTrainPairs)
        print("Source Language Vocabulary Length (Number of Encoder Tokens)  :", self.numEncoderTokens)
        print("Target Language Vocabulary length (Number of Decoder Tokens)  :", self.numDecoderTokens)
        print("Max sequence length for inputs (Max Source Lang Word Length)  :", self.maxSourceLength)
        print("Max sequence length for outputs (Max Source Lang Word Length) :", self.maxTargetLength)

        # Create the required dictionaries and store them in sourceVocab and targetVocab respectively
        sourceVocab = self.dictionaryLookup(sourceChars)
        targetVocab = self.dictionaryLookup(targetChars)

        return sourceVocab, targetVocab

    def tensorFromWord(self, charToInt, word, language) :

        ''' Input ->    charToInt : Dictionary that contains the mapping of character to its unique integer.
                        word : Word of which the tensor is to be created.    
                        language : flag to check if "source" or "target" language

            Output ->   Creates a list containing unique integer corresponding to each character in word, 
                        appends EOW2Int, 
                        appends PAD2Int tokens (if necessary),
                        converts it to tensor,
                        and stores it on the device specified. '''
        
        indexes = []
        
        # List of unique integer corresponding to each character
        for char in word :
            if char in charToInt :
                indexes.append(charToInt[char])
            else :
                indexes.append(self.UNK2Int)

        # We represent End of Word with EOW2Int
        indexes.append(self.EOW2Int)

        # Add Padding after EOW 
        if language == "source" :
            indexes.extend([self.PAD2Int for i in range(self.maxSourceLength - len(indexes) + 1)])

        else :
            indexes.extend([self.PAD2Int for i in range(self.maxTargetLength - len(indexes) + 1)])

        return torch.tensor(indexes, dtype = torch.long, device = self.device).view(-1, 1)

    def tensorsFromPair(self, pairOfWords) :
        
        ''' Input ->    pairOfWords : A pair of source language word and target language word.    

            Output ->   Creates a tensor for source language word and target language word
                        Returns a pair of the two tensors created. '''

        sourceTensor = self.tensorFromWord(self.sourceCharToInt, pairOfWords[0], "source")
        targetTensor = self.tensorFromWord(self.targetCharToInt, pairOfWords[1], "target")

        return (sourceTensor, targetTensor)

    def createTrainPairs(self) :

        ''' Input ->    N/A   
            Output ->   Returns a List of Tuples where each tuple contains a source word and its corresponding target word. '''
        
        return list(zip(self.train["source"].to_list(), self.train["target"].to_list()))
    
    def createValidPairs(self) :

        ''' Input ->    N/A   
            Output ->   Returns a List of Tuples where each tuple contains a source word and its corresponding target word. '''
        
        return list(zip(self.valid["source"].to_list(), self.valid["target"].to_list()))
    
    def createTestPairs(self) :

        ''' Input ->    N/A   
            Output ->   Returns a List of Tuples where each tuple contains a source word and its corresponding target word. '''
        
        return list(zip(self.test["source"].to_list(), self.test["target"].to_list()))
    
    def getMaxLength(self) :
        
        ''' Returns the maximum of sourceMaxLength and targetMaxLength '''

        return max(self.maxSourceLength, self.maxTargetLength)


if __name__ == "__main__":
    dataProcessor = DataProcessing(DATAPATH = 'aksharantar_sampled', targetLanguage = 'hin')
