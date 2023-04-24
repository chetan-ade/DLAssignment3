''' Import Required Packages. '''

from __future__ import unicode_literals, print_function, division
import torch

''' Import files with User Defined Classes. '''

import model
from dataProcessing import DataProcessing
from trainModel import Training

if __name__ == "__main__" :

    # Create a configuration dictionary
    configs = {

        'device'                    : torch.device("cuda" if torch.cuda.is_available() else "cpu"),     # Set device to gpu if available, else set device to cpu 
        'hiddenSize'                : 128,                                                              # Hidden Size in RNN Layer
        'cellType'                  : "LSTM",                                                           # Cell Type = ['RNN', 'GRU', 'LSTM]
        'embeddingSize'             : 128,                                                              # Embedding Size for each character
        'numLayersEncoderDecoder'   : 2,                                                                # Number of RNN Layers in Encoder / Decoder
        'dropout'                   : 0,                                                                # Dropout Probability

    }

    # Pre-Process the data
    dataProcessor = DataProcessing(DATAPATH = 'aksharantar_sampled', targetLanguage = 'hin', configs = configs)
    
    # Create an encoder object with inputSize = number of characters in source language and hiddenSize
    encoder = model.Encoder(inputSize = dataProcessor.numEncoderTokens, configs = configs).to(configs['device'])
    
    # Create a decoder object with hiddenSie and outputSize = number of characters in target language
    decoder = model.Decoder(outputSize = dataProcessor.numDecoderTokens, configs = configs).to(configs['device'])
    
    # Create a model Training object with the data Processor
    modelTraining = Training(dataProcessor)
    
    # Train the Encoder Decoder Model 
    modelTraining.trainIters(encoder, decoder, nIters = dataProcessor.numTrainPairs // 50, printEvery = dataProcessor.numTrainPairs // 500, plotEvery = dataProcessor.numTrainPairs // 5000)