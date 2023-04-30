''' Import Required Packages. '''

from __future__ import unicode_literals, print_function, division
import torch
import time

''' Import files with User Defined Classes. '''

import model
from dataProcessing import DataProcessing
from trainModel import Training

def trainForConfigs(configs, dataProcessor) :

    ''' Trains Encoder Decoder Model for given configs and processed data. '''
    
    print('CONFIGS')    # Print the configurations
    for parameters in configs.items() :
        print(parameters)

    configs['maxLengthWord'] = dataProcessor.maxLengthWord # Store max length in configs for later use

    dataProcessor.updateConfigs(configs) # Update configs in data processor
    
    encoder = model.Encoder(inputSize = dataProcessor.numEncoderTokens, configs = configs).to(configs['device']) # Create an encoder object
    
    # Create a decoder object
    if configs['attention'] :
        decoder = model.DecoderAttention(outputSize = dataProcessor.numDecoderTokens, configs = configs).to(configs['device'])
    else :
        decoder = model.Decoder(outputSize = dataProcessor.numDecoderTokens, configs = configs).to(configs['device'])
    
    modelTraining = Training(dataProcessor) # Create a modelTraining object
    
    startTime = time.time()

    modelTraining.train(encoder, decoder) # Train the Encoder Decoder Model 

    print("TIME : ", (time.time() - startTime) / 60)

def trainConfigsGrid(device, dataProcessor) :
    
    f = open("log.txt", 'w') # Write Status of Configs in a File for Debugging

    for cell in ['LSTM', 'RNN', 'GRU'] :
        for attentionFlag in [False] :
            for bidirectional in [True, False] :
                for numLayers in [1, 3] :

                    try :

                        configs = { # Create a configuration dictionary
        
                            'device'                    : device,  # Available Device (CPU / CUDA) 
                            'hiddenSize'                : 64,     # Hidden Size in RNN Layer
                            'cellType'                  : cell,   # Cell Type = ['RNN', 'GRU', 'LSTM']
                            'embeddingSize'             : 256,     # Embedding Size for each character
                            'numLayersEncoderDecoder'   : numLayers,       # Number of RNN Layers in Encoder / Decoder
                            'dropout'                   : 0,       # Dropout Probability
                            'attention'                 : attentionFlag,   # True = AttentionDecoder, False = Decoder
                            'batchSize'                 : 512,      # Batch Size for Training and Evaluating
                            'epochs'                    : 2,       # Total Number of Training Epochs
                            'bidirectional'             : bidirectional,
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
    
        'device'                    : device,  # Available Device (CPU / CUDA) 
        'hiddenSize'                : 64,     # Hidden Size in RNN Layer
        'cellType'                  : "LSTM",   # Cell Type = ['RNN', 'GRU', 'LSTM']
        'embeddingSize'             : 256,     # Embedding Size for each character
        'numLayersEncoderDecoder'   : 3,       # Number of RNN Layers in Encoder / Decoder
        'dropout'                   : 0,       # Dropout Probability
        'attention'                 : False,   # True = AttentionDecoder, False = Decoder
        'batchSize'                 : 512,      # Batch Size for Training and Evaluating
        'epochs'                    : 2,       # Total Number of Training Epochs
        'bidirectional'             : True,
        'learningRate'              : 0.001

    }

    trainForConfigs(configs, dataProcessor) # Train for configurations
                    
    



    