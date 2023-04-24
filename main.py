''' Import Required Packages. '''

from __future__ import unicode_literals, print_function, division
import torch

''' Import files with User Defined Classes. '''

import model
from dataProcessing import DataProcessing
from trainModel import Training

if __name__ == "__main__" :

    # Set device to gpu if available, else set device to cpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hidden Size # HyperParameter
    hiddenSize = 256

    # RNN Type # HyperParameter
    RNNType = "GRU"

    # Pre-Process the data
    dataProcessor = DataProcessing(DATAPATH = 'aksharantar_sampled', targetLanguage = 'hin', device = device)
    
    # Create an encoder object with inputSize = number of characters in source language and hiddenSize
    encoder = model.Encoder(inputSize = dataProcessor.numEncoderTokens, hiddenSize = hiddenSize, device = device, RNNType = RNNType).to(device)
    
    # Create a decoder object with hiddenSie and outputSize = number of characters in target language
    decoder = model.Decoder(hiddenSize, dataProcessor.numDecoderTokens, device = device, RNNType = RNNType).to(device)
    
    # Create a model Training object with the data Processor
    modelTraining = Training(dataProcessor)
    
    # Train the Encoder Decoder Model 
    modelTraining.trainIters(encoder, decoder, nIters = dataProcessor.numTrainPairs // 50, printEvery = dataProcessor.numTrainPairs // 500, plotEvery = dataProcessor.numTrainPairs // 5000)