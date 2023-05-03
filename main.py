''' Import Required Packages. '''

from __future__ import unicode_literals, print_function, division
import torch

''' Import files with User Defined Classes. '''

import model
from dataProcessing import DataProcessing
from trainModel import Training

def trainForConfigs(configs, dataProcessor) :

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

    modelTraining.train() # Train the Encoder Decoder Model 

def trainConfigsGrid(device, dataProcessor) :

    ''' Trains model for all mentioned combinations '''
    
    f = open("log.txt", 'w') # Write Status of Configs in a File for Debugging

    for cell in ['LSTM', 'RNN', 'GRU'] :
        for attentionFlag in [False] :
            for bidirectional in [True, False] :
                for numLayers in [1, 3] :

                    try :

                        configs = { # Create a configuration dictionary
    
                            'device'                    : device,  # Available Device (CPU / CUDA) 
                            'hiddenSize'                : 256,     # Hidden Size in RNN Layer
                            'cellType'                  : "LSTM",   # Cell Type = ['RNN', 'GRU', 'LSTM']
                            'embeddingSize'             : 256,     # Embedding Size for each character
                            'numLayersEncoderDecoder'   : 2,       # Number of RNN Layers in Encoder / Decoder
                            'dropout'                   : 0.2,       # Dropout Probability
                            'attention'                 : False,   # True = AttentionDecoder, False = Decoder
                            'batchSize'                 : 32,      # Batch Size for Training and Evaluating
                            'epochs'                    : 10,       # Total Number of Training Epochs
                            'bidirectional'             : True,
                            'learningRate'              : 0.001

                        }

                        trainForConfigs(configs, dataProcessor) # Train for configurations
                        f.write('SUCCESS : ' + str(configs) + "\n")

                    except Exception :
                        
                        f.write('Error in CONFIGS : ' + str(configs) + "\n") # Write Configs in File
                 
    f.close()
    
if __name__ == "__main__" :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProcessor = DataProcessing('aksharantar_sampled', 'hin', device) # Pre-Process the data

    configs = { # Create a configuration dictionary
    
        'device'                    : device,   # Available Device (CPU / CUDA) 
        'hiddenSize'                : 64,      # Hidden Size in RNN Layer
        'cellType'                  : "GRU",   # Cell Type = ['RNN', 'GRU', 'LSTM']
        'embeddingSize'             : 256,      # Embedding Size for each character
        'numLayersEncoderDecoder'   : 2,        # Number of RNN Layers in Encoder / Decoder
        'dropout'                   : 0.2,      # Dropout Probability
        'attention'                 : True,    # True = AttentionDecoder, False = Decoder
        'batchSize'                 : 32,       # Batch Size for Training and Evaluating
        'epochs'                    : 10,       # Total Number of Training Epochs
        'bidirectional'             : False,     # Bidirectional Flag
        'learningRate'              : 0.001     # Learning Rate for Optimizers

    }

    trainForConfigs(configs, dataProcessor) # Train for configurations
                    
    



    