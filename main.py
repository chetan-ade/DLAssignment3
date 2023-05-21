''' Import Required Packages. '''

from __future__ import unicode_literals, print_function, division
import torch
import argparse

''' Import files with User Defined Classes. '''

import model
from dataProcessing import DataProcessing
from trainModel import Training

def updateConfigs(configs):

        parser = argparse.ArgumentParser(description='DLAssignment3 Parser')

        parser.add_argument('-wp', '--wandb_project',
                            type=str, metavar='', help='wandb project')
        
        parser.add_argument('-we', '--wandb_entity', type=str,
                            metavar='', help='wandb entity')
        
        parser.add_argument('-hs', '--hidden_size', type=int,
                            metavar='', help='hidden_size')
        
        parser.add_argument('-c', '--cell_type', type=str,
                            metavar='', help='cell_type')
        
        parser.add_argument('-nle', '--numLayersEncoder', type=int,
                            metavar='', help='numLayersEncoder')
        
        parser.add_argument('-nld', '--numLayersDecoder', type=int,
                            metavar='', help='numLayersDecoder')
        
        parser.add_argument('-dp', '--drop_out', type=float,
                            metavar='', help='drop_out')
        
        parser.add_argument('-es', '--embedding_size',
                            type=int, metavar='', help='embedding_size')
        
        parser.add_argument('-bs', '--batch_size',
                            type=int, metavar='', help='batch_size')
        
        parser.add_argument('-e', '--epoch',
                            type=int, metavar='', help='epoch')
        
        parser.add_argument('-lr', '--learning_rate',
                            type=float, metavar='', help='learning rate')

        parser.add_argument('-a', '--attention',
                            type=int, metavar='', help='attention')

        parser.add_argument('-bd', '--bidirectional',
                            type=int, metavar='', help='bidirectional')
        
        args = parser.parse_args()

        if (args.wandb_project != None):
            configs["wandb_project"] = args.wandb_project
        
        if (args.wandb_entity != None):
            configs["wandb_entity"] = args.wandb_entity
        
        if (args.hidden_size != None):
            configs["hiddenSize"] = args.hidden_size
        
        if (args.cell_type != None):
            configs["cellType"] = args.cell_type
        
        if (args.numLayersEncoder != None):
            configs["numLayersEncoder"] = args.numLayers

        if (args.numLayersDecoder != None):
            configs["numLayersDecoder"] = args.numLayers   
        
        if (args.drop_out != None):
            configs["dropout"] = args.drop_out
        
        if (args.embedding_size != None):
            configs["embeddingSize"] = args.embedding_size
        
        if (args.batch_size != None):
            configs["batchSize"] = args.batch_size
        
        if (args.epoch != None):
            configs["epochs"] = args.epoch
        
        if (args.learning_rate != None):
            configs["learningRate"] = args.learning_rate

        if (args.attention != None):
            if (args.attention == 0):
                configs["attention"] = False
            else:
                configs['attention'] = True

        if (args.bidirectional != None):
            if (args.bidirectional == 0):
                configs["bidirectional"] = False
            else:
                configs['bidirectional'] = True

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
    # modelTraining.evaluateTest()

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

    updateConfigs(configs)
    trainForConfigs(configs, dataProcessor, fromMain = True) # Train for configurations
                    
    



    