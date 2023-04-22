from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):

    # Encoder Constructor
    def __init__(self, inputSize, hiddenSize, device) :

        ''' The encoder of a seq2seq network is a RNN that outputs some value for every character from the input word. 
            For every input character the encoder outputs a vector and a hidden state, and uses the hidden state for the next input character.
            
            INPUT :     inputSize : Number of Characters in Source Language.
                        hiddenSize : Size of embedding for each character,
                                     Size of Input for GRU / RNN / LSTM,
                                     Size of Hidden State for GRU / RNN / LSTM,
                                     Size of Input for Dense Layer.
  
            OUTPUT :    Encoder Object '''

        # Call the constructor for NN Module
        super(EncoderRNN, self).__init__()

        # Store the parameters in class variables
        self.hiddenSize = hiddenSize

        # Create an Embedding for the Input # Each character will have an embedding of size = hiddenSize
        self.embedding = nn.Embedding(num_embeddings = inputSize, embedding_dim = hiddenSize)

        # The RNN / LSTM / GRU Layer # Since the input is embedded input, we have the first parameter as hiddenSize # We are setting the size of hidden state also as hiddenSize
        self.gru = nn.GRU(hiddenSize, hiddenSize)

        # Store device
        self.device = device

    # Encoder Forward Pass
    def forward(self, input, hidden):

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        embedded = self.embedding(input).view(1, 1, -1)

        # Pass the embedded input to the RNN / GRU / LSTM Layer
        output = embedded

        # GRU takes as input 
        #   1. Tensor of Shape (L, Hin) for un-batched input or Tensor of Shape (L, N, Hin) for batch of size N 
        #   2. Tensor of Shape (D * num_layers, H out) for un-batched input or Tensor of Shape (D * num_layers, N, H out) for batch of size N 
        # GRU gives output
        #   1. Tensor of shape (L, D * H out) for un-batched input or Tensor of Shape (L, N, D * H out) for batch of size N
        #      It contains the output features (h_t) from the last layer of GRU, for each t.
        #   2. hidden h_n : tensor of shape (D * num_layers, H out) for un-batched input or Tensor of Shape (D * num_layers, N, H out) for batch of size 
        #      It contains the final hidden state for the input sequence 
        output, hidden = self.gru(output, hidden)

        # Return the output of RNN / GRU / LSTM Layer   
        return output, hidden

    # Encoder Hidden State Initialization
    def initHidden(self):

        # Returns a tensor of shape (1, 1, hiddenSize) and stores it on device # It is used while training for initialization
        return torch.zeros(1, 1, self.hiddenSize, device = self.device)
    
class DecoderRNN(nn.Module):

    # Decoder Constructor
    def __init__(self, hiddenSize, output_size, device) :

        ''' 
            INPUT :     outputSize : Number of Characters in Target Language.
                        hiddenSize : Size of embedding for each character,
                                     Size of Input for GRU / RNN / LSTM,
                                     Size of Hidden State for GRU / RNN / LSTM,
                                     Size of Input for Dense Layer.

            OUTPUT :    Decoder Object '''

        # Call the constructor for NN Module
        super(DecoderRNN, self).__init__()

        # Store the parameters in class variables 
        self.hiddenSize = hiddenSize

        # Create an Embedding for the Input
        self.embedding = nn.Embedding(num_embeddings = output_size, embedding_dim = hiddenSize)

        # The RNN / LSTM / GRU Layer
        self.gru = nn.GRU(hiddenSize, hiddenSize)

        # Linear layer that will take GRU / RNN / LSTM output as input
        self.out = nn.Linear(hiddenSize, output_size)

        # SoftMax Layer for the final output
        self.softmax = nn.LogSoftmax(dim = 1)

        # Store device
        self.device = device

    # Decoder Forward Pass
    def forward(self, input, hidden) :

        # Pass the Input through the Embedding layer to get embedded input # The embedded input is reshaped to have a shape of (1, 1, -1)
        output = self.embedding(input).view(1, 1, -1)

        # Pass the embedded input through relu
        output = F.relu(output)

        # Pass the output of relu and the previous hidden state to GRU
        output, hidden = self.gru(output, hidden)

        # Pass the 0th Output through Linear Layer and then through SoftMax Layer # ? Why the 0th Layer? Is it the final layer?
        output = self.softmax(self.out(output[0]))

        # Return the output and hidden state
        return output, hidden