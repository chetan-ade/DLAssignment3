from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module) :

    # Encoder Constructor
    def __init__(self, inputSize, configs) :

        super(Encoder, self).__init__() # Call the constructor for NN Module
        
        # Store the parameters in class variables
        self.hiddenSize = configs['hiddenSize']
        self.embeddingSize = configs['embeddingSize']
        self.cellType = configs['cellType']
        self.device = configs['device']
        self.numLayersEncoderDecoder = configs['numLayersEncoderDecoder']
        self.dropout = configs['dropout']
        self.batchSize = configs['batchSize']
        self.bidirectional = configs['bidirectional']

        self.embedding = nn.Embedding(num_embeddings = inputSize, embedding_dim = self.embeddingSize) # Create an Embedding for the Input # Each character will have an embedding of size = hiddenSize
        
        # Create cell layer
        if self.cellType == 'GRU' :
            self.RNNLayer = nn.GRU(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout, bidirectional = self.bidirectional)

        elif self.cellType == 'RNN' : 
            self.RNNLayer = nn.RNN(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout, bidirectional = self.bidirectional)

        elif self.cellType == 'LSTM': 
            self.RNNLayer = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout, bidirectional = self.bidirectional)

    # Encoder Forward Pass
    def forward(self, input, hidden) :

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        embedded = self.embedding(input).view(1, self.batchSize, -1)
        output = embedded
        output, hidden = self.RNNLayer(output, hidden) # Pass the embedded input to the RNN / GRU Layer
        
        return output, hidden

    # Encoder Hidden State Initialization
    def initHidden(self) :

        if self.bidirectional :
            return torch.zeros(self.numLayersEncoderDecoder * 2, self.batchSize, self.hiddenSize, device = self.device)
        
        else :
            return torch.zeros(self.numLayersEncoderDecoder, self.batchSize, self.hiddenSize, device = self.device)
    
    # Encoder Hidden Cell Initialization
    def initCell(self) :

        if self.bidirectional :

            # Returns a tensor of shape (1, 1, hiddenSize) and stores it on device # It is used while training for initialization
            return torch.zeros(self.numLayersEncoderDecoder * 2, self.batchSize, self.hiddenSize, device = self.device)
        
        else :

            # Returns a tensor of shape (1, 1, hiddenSize) and stores it on device # It is used while training for initialization
            return torch.zeros(self.numLayersEncoderDecoder, self.batchSize, self.hiddenSize, device = self.device)
    
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
        self.batchSize = configs['batchSize']
        self.bidirectional = configs['bidirectional']

        # Create an Embedding for the Input
        self.embedding = nn.Embedding(num_embeddings = outputSize, embedding_dim = self.embeddingSize)

        # The RNN / LSTM / GRU Layer
        if self.cellType == 'GRU' :
            self.RNNLayer = nn.GRU(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout, bidirectional = self.bidirectional)

        elif self.cellType == 'RNN' :
            self.RNNLayer = nn.RNN(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout, bidirectional = self.bidirectional)
        
        else : 
            self.RNNLayer = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropout, bidirectional = self.bidirectional)

        # Linear layer that will take GRU / RNN / LSTM output as input
        if self.bidirectional:
            self.out = nn.Linear(2 * self.hiddenSize, outputSize)

        else :
            self.out = nn.Linear(self.hiddenSize, outputSize)

        # SoftMax Layer for the final output
        self.softmax = nn.LogSoftmax(dim = 1)

    # Decoder Forward Pass
    def forward(self, input, hidden) :

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        output = self.embedding(input).view(1, self.batchSize, -1)

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
        self.batchSize = configs['batchSize']
        self.bidirectional = configs['bidirectional']

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
            self.RNNLayer = nn.GRU(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropoutRate, bidirectional = self.bidirectional)

        elif self.cellType == 'RNN' :
            self.RNNLayer = nn.RNN(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropoutRate, bidirectional = self.bidirectional)
        
        else : 
            self.RNNLayer = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers = self.numLayersEncoderDecoder, dropout = self.dropoutRate, bidirectional = self.bidirectional)

        # Linear layer that will take GRU / RNN / LSTM output as input
        if self.bidirectional:
            self.out = nn.Linear(2 * self.hiddenSize, outputSize)

        else :
            self.out = nn.Linear(self.hiddenSize, outputSize)

    # Decoder Forward Pass
    def forward(self, input, hidden, encoder_outputs) :

        # print("INPUT :", input.shape)
        # print("HIDDEN[0] :", hidden[0].shape)
        # print("ENC_OP :", encoder_outputs.shape)

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        embedded = self.embedding(input).view(1, self.batchSize, -1)

        # Dropout embedded layer according to dropout probability
        embedded = self.dropout(embedded)

        # print("EMBEDDED : ", embedded.shape)
        # print("EMBEDDED[0] : ", embedded[0].shape)
        # print("HIDDEN[0][0] : ", hidden[0][0].shape)
        

        if self.cellType == 'LSTM' :

            # Calculate Attention Weights

            embeddedHidden = torch.cat((embedded[0], hidden[0][0]), 1)
            # print("EMBEDDED HIDDEN : ", embeddedHidden.shape)

            embeddedHiddenAttention = self.attentionLayer(embeddedHidden)
            # print("EMBEDDED HIDDEN ATTENTION : ", embeddedHiddenAttention.shape)

            attentionWeights = F.softmax(embeddedHiddenAttention, dim = 1)
        
        else :

            # Calculate Attention Weights
            attentionWeights = F.softmax(self.attentionLayer(torch.cat((embedded[0], hidden[0]), 1)), dim = 1)

        
        # Batch matrix-matrix product of attention weights with encoder outputs
        # print(attentionWeights.shape)
        # print(encoder_outputs.shape)

        # print(attentionWeights.unsqueeze(0).shape)
        # print(encoder_outputs.unsqueeze(0).shape)
        
        # attentionApplied = torch.bmm(attentionWeights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        attentionApplied = torch.bmm(attentionWeights.view(self.batchSize, 1, self.maxLengthTensor), encoder_outputs).view(1, self.batchSize, -1)

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