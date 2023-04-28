from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module) :

    # Encoder Constructor
    def __init__(self, inputSize, configs) :

        ''' The encoder of a seq2seq network is a RNN that outputs some value for every character from the input word. 
            For every input character the encoder outputs a vector and a hidden state, and uses the hidden state for the next input character.
            
            INPUT :     inputSize  : Number of Characters in Source Language.
                        hiddenSize : Size of embedding for each character,
                                     Size of Input for GRU / RNN / LSTM,
                                     Size of Hidden State for GRU / RNN / LSTM,
                                     Size of Input for Dense Layer.
                        device     : device on which tensors are stored
                        cellType    : RNN / GRU / LSTM 
  
            OUTPUT :    Encoder Object '''

        # Call the constructor for NN Module
        super(Encoder, self).__init__()

        # Store the parameters in class variables
        self.hiddenSize = configs['hiddenSize']
        self.embeddingSize = configs['embeddingSize']
        self.cellType = configs['cellType']
        self.device = configs['device']
        self.numLayersEncoderDecoder = configs['numLayersEncoderDecoder']
        self.dropout = configs['dropout']

        # Create an Embedding for the Input # Each character will have an embedding of size = hiddenSize
        self.embedding = nn.Embedding(num_embeddings = inputSize, embedding_dim = self.embeddingSize)

        # The RNN / LSTM / GRU Layer # Since the input is embedded input, we have the first parameter as hiddenSize # We are setting the size of hidden state also as hiddenSize
        if self.cellType == 'GRU' :
            self.RNNLayer = nn.GRU(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout)

        elif self.cellType == 'RNN' : 
            self.RNNLayer = nn.RNN(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout)

        else : 
            self.RNNLayer = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout)

    # Encoder Forward Pass
    def forward(self, input, hidden) :

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        # Pass the embedded input to the RNN / GRU Layer
        output, hidden = self.RNNLayer(output, hidden)

        # Return the output of RNN / GRU Layer   
        return output, hidden

    # Encoder Hidden State Initialization
    def initHidden(self) :

        # Returns a tensor of shape (1, 1, hiddenSize) and stores it on device # It is used while training for initialization
        return torch.zeros(self.numLayersEncoderDecoder, 1, self.hiddenSize, device = self.device)
    
    # Encoder Hidden Cell Initialization
    def initCell(self) :

        # Returns a tensor of shape (1, 1, hiddenSize) and stores it on device # It is used while training for initialization
        return torch.zeros(self.numLayersEncoderDecoder, 1, self.hiddenSize, device = self.device)
    
class Decoder(nn.Module) :

    # Decoder Constructor
    def __init__(self, outputSize, configs) :

        ''' 
            INPUT :     outputSize : Number of Characters in Target Language.
                        hiddenSize : Size of embedding for each character,
                                     Size of Input for GRU / RNN / LSTM,
                                     Size of Hidden State for GRU / RNN / LSTM,
                                     Size of Input for Dense Layer.

            OUTPUT :    Decoder Object '''

        # Call the constructor for NN Module
        super(Decoder, self).__init__()

        # Store the parameters in class variables 
        self.hiddenSize = configs['hiddenSize']
        self.embeddingSize = configs['embeddingSize']
        self.cellType = configs['cellType']
        self.device = configs['device'] 
        self.numLayersEncoderDecoder = configs['numLayersEncoderDecoder']
        self.dropout = configs['dropout']

        # Create an Embedding for the Input
        self.embedding = nn.Embedding(num_embeddings = outputSize, embedding_dim = self.embeddingSize)

        # The RNN / LSTM / GRU Layer
        if self.cellType == 'GRU' :
            self.RNNLayer = nn.GRU(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout)

        elif self.cellType == 'RNN' :
            self.RNNLayer = nn.RNN(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout)
        
        else : 
            self.RNNLayer = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout)

        # Linear layer that will take GRU / RNN / LSTM output as input
        self.out = nn.Linear(self.hiddenSize, outputSize)

        # SoftMax Layer for the final output
        self.softmax = nn.LogSoftmax(dim = 1)

    # Decoder Forward Pass
    def forward(self, input, hidden) :

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        output = self.embedding(input).view(1, 1, -1)

        # Pass the embedded input through relu
        output = F.relu(output)

        # Pass the output of relu and the previous hidden state to GRU / RNN
        output, hidden = self.RNNLayer(output, hidden)

        # Pass the 0th Output through Linear Layer and then through SoftMax Layer # ? Why the 0th Layer? Is it the final layer?
        output = self.softmax(self.out(output[0]))

        # Return the output and hidden state
        return output, hidden
    
class DecoderAttention(nn.Module) :

    # Decoder Constructor
    def __init__(self, outputSize, configs) :

        ''' 
            INPUT :     outputSize : Number of Characters in Target Language.
                        hiddenSize : Size of embedding for each character,
                                     Size of Input for GRU / RNN / LSTM,
                                     Size of Hidden State for GRU / RNN / LSTM,
                                     Size of Input for Dense Layer.

            OUTPUT :    Decoder Object '''

        # Call the constructor for NN Module
        super(DecoderAttention, self).__init__()

        # Store the parameters in class variables 
        self.hiddenSize = configs['hiddenSize']
        self.embeddingSize = configs['embeddingSize']
        self.cellType = configs['cellType']
        self.device = configs['device'] 
        self.numLayersEncoderDecoder = configs['numLayersEncoderDecoder']
        self.dropoutRate = configs['dropout']
        self.maxLengthWord = configs['maxLengthWord']
        self.maxLengthTensor = self.maxLengthWord + 1

        # Create an Embedding for the Input
        self.embedding = nn.Embedding(num_embeddings = outputSize, embedding_dim = self.embeddingSize)

        # Attention Layer
        self.attentionLayer = nn.Linear(self.embeddingSize + self.hiddenSize, self.maxLengthTensor)

        # Combine Embedded and Attention Applied Outputs
        self.attentionCombine = nn.Linear(self.embeddingSize + self.hiddenSize, self.embeddingSize)

        # Dropout Layer
        self.dropout = nn.Dropout(self.dropoutRate)

        # The RNN / LSTM / GRU Layer
        if self.cellType == 'GRU' :
            self.RNNLayer = nn.GRU(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropoutRate)

        elif self.cellType == 'RNN' :
            self.RNNLayer = nn.RNN(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropoutRate)
        
        else : 
            self.RNNLayer = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropoutRate)

        # Linear layer that will take GRU / RNN / LSTM output as input
        self.out = nn.Linear(self.hiddenSize, outputSize)

    # Decoder Forward Pass
    def forward(self, input, hidden, encoder_outputs) :

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        embedded = self.embedding(input).view(1, 1, -1)

        # Dropout embedded layer according to dropout probability
        embedded = self.dropout(embedded)

        if self.cellType == 'LSTM' :

            # Calculate Attention Weights
            attentionWeights = F.softmax(self.attentionLayer(torch.cat((embedded[0], hidden[0][0]), 1)), dim = 1)
        
        else :

            # Calculate Attention Weights
            attentionWeights = F.softmax(self.attentionLayer(torch.cat((embedded[0], hidden[0]), 1)), dim = 1)

        # Batch matrix-matrix product of attention weights with encoder outputs
        attentionApplied = torch.bmm(attentionWeights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # Concatenate Embedded Output and product of batch matrix matrix multiplication
        output = torch.cat((embedded[0], attentionApplied[0]), 1)

        # Pass the output through the attention combine layer
        output = self.attentionCombine(output).unsqueeze(0)

        # Apply ReLU activation
        output = F.relu(output)

        # Pass the output of ReLU layer from RNN Layer
        output, hidden = self.RNNLayer(output, hidden)
        
        # Apply softmax to the output of out linear layer
        output = F.log_softmax(self.out(output[0]), dim = 1)

        # Return the output, hidden state and attention weights
        return output, hidden, attentionWeights