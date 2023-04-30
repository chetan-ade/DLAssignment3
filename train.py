''' Import Required Packages. '''

import wandb
import torch

''' Import files with User Defined Classes. '''

from dataProcessing import DataProcessing
import main

if __name__ == "__main__" :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProcessor = DataProcessing('aksharantar_sampled', 'hin', device) # Pre-Process the data

    # main.trainConfigsGrid(device, dataProcessor)

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

    main.trainForConfigs(configs, dataProcessor) # Train for configurations