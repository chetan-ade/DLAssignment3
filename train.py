""" Import Required Packages. """

import wandb
import torch

""" Import files with User Defined Classes. """

import main
from dataProcessing import DataProcessing

def sweepTrain() :

    ''' Updates configs and trains wandb sweeps. '''

    wandb.init(config = sweepConfigs)

    # Update configs using wandb sweep configs
    configs["hiddenSize"]               = wandb.config.hidden_size
    configs["cellType"]                 = wandb.config.cell_type
    configs["numLayersEncoderDecoder"]  = wandb.config.num_layers
    configs["dropout"]                  = wandb.config.drop_out
    configs["embeddingSize"]            = wandb.config.embedding_size
    configs["bidirectional"]            = wandb.config.bidirectional
    configs["batchSize"]                = wandb.config.batch_size
    configs["epochs"]                   = wandb.config.epoch
    configs["learningRate"]             = wandb.config.learning_rate
    configs["attention"]                = wandb.config.attention

    wandb.run.name = ( # Wandb Run Name
        "cell_type_"
        + str(wandb.config.cell_type)
        + "_numLayers_"
        + str(wandb.config.num_layers)
        + "_attention_"
        + str(wandb.config.attention)
    )

    main.trainForConfigs(configs, dataProcessor)  # Train for configurations

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProcessor = DataProcessing('aksharantar_sampled', 'hin', device) # Pre-Process the data  

    configs = {  # Create a configuration dictionary
        
        'wandb_project'             : 'DL Assignment 3' ,   # Wandb Credentials
        'wandb_entity'              : 'cs22m033'        ,           
        'device'                    : device            ,   # Available Device (CPU / CUDA)
        'hiddenSize'                : 64                ,   # Hidden Size in Cell Layer
        'cellType'                  : 'LSTM'            ,   # Cell Type = ['RNN', 'GRU', 'LSTM']
        'embeddingSize'             : 256               ,   # Embedding Size for each character
        'numLayersEncoderDecoder'   : 3                 ,   # Number of RNN Layers in Encoder / Decoder
        'dropout'                   : 0                 ,   # Dropout Probability
        'attention'                 : False             ,   # True = AttentionDecoder, False = Decoder
        'batchSize'                 : 512               ,   # Batch Size for Training and Evaluating
        'epochs'                    : 2                 ,   # Total Number of Training Epochs
        'bidirectional'             : True              ,   # Bidirectional Flag
        'learningRate'              : 0.001             ,   # Optimizer learning rate

    }

    sweepConfigs = {  # Sweep Configs
        
        'method': 'bayes',
        
        'name': 'bayes_sweep',

        'metric': {
            'name': 'validation accuracy', 'goal': 'maximize'
        },

        'parameters' : {
            'epoch': {
                'values': [10, 15]
            },
            
            'hidden_size': {
                'values': [64, 128, 256]
            },
            
            'cell_type': {
                'values': ['LSTM', 'RNN', 'GRU']
            },
            
            'learning_rate': {
                'values': [1e-2, 1e-3]
            },
            
            'num_layers': {
                'values': [1, 2, 3]
            },
            
            'drop_out': {
                'values': [0.0, 0.2, 0.3]
            },
            
            'embedding_size': {
                'values': [64, 128, 256]
            },
            
            'batch_size': {
                'values': [32, 64, 128]
            },
            
            'bidirectional': {
                'values': [True, False]
            },
            
            'attention': {
                'values': [False]
            },
        }
        
        
    }

    sweepId = wandb.sweep(sweep = sweepConfigs, project = configs["wandb_project"])
    wandb.agent(sweep_id = sweepId, function = sweepTrain)  # Start agent
