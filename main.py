''' Import Required Packages. '''

from __future__ import unicode_literals, print_function, division
import torch

''' Import files with User Defined Classes. '''

import model
from dataProcessing import DataProcessing
from trainModel import Training

def trainForConfigs(configs, dataProcessor, fromMain = False) :

    ''' Train Model for given configs and processed data. '''

    for parameter in configs.items() :
        print(parameter) # Print the configurations

    configs['maxLengthWord'] = dataProcessor.maxLengthWord # Store max length in configs for later use

    dataProcessor.updateConfigs(configs) # Update configs in data processor
    
    encoder = model.Encoder(inputSize = dataProcessor.numEncoderTokens, configs = configs).to(configs['device']) # Create an encoder object
    
    # Create a decoder object
    if configs['attention'] :
        decoder = model.DecoderAttention(outputSize = dataProcessor.numDecoderTokens, configs = configs).to(configs['device'])
    else :
        decoder = model.Decoder(outputSize = dataProcessor.numDecoderTokens, configs = configs).to(configs['device'])
    
    modelTraining = Training(dataProcessor, encoder, decoder) # Create a modelTraining object

    modelTraining.train(fromMain) # Train the Encoder Decoder Model 

    # Test Data - Predictions
    modelTraining.evaluateTest()

if __name__ == "__main__" :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProcessor = DataProcessing('aksharantar_sampled', 'hin', device) # Pre-Process the data

    configs = { # Create a configuration dictionary
    
        'device'                    : device,   # Available Device (CPU / CUDA) 
        'hiddenSize'                : 512,      # Hidden Size in RNN Layer
        'cellType'                  : "LSTM",   # Cell Type = ['RNN', 'GRU', 'LSTM']
        'embeddingSize'             : 256,      # Embedding Size for each character
        'numLayersEncoder'          : 3,        # Number of RNN Layers in Encoder / Decoder
        'numLayersDecoder'          : 1,
        'dropout'                   : 0.2,      # Dropout Probability
        'attention'                 : False,    # True = AttentionDecoder, False = Decoder
        'batchSize'                 : 64,       # Batch Size for Training and Evaluating
        'epochs'                    : 15,       # Total Number of Training Epochs
        'bidirectional'             : True,     # Bidirectional Flag
        'learningRate'              : 0.001,     # Learning Rate for Optimizers
        'debug'                     : False

    }

    trainForConfigs(configs, dataProcessor, fromMain = True) # Train for configurations
                    
    



    