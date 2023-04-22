from __future__ import unicode_literals, print_function, division
import torch
import random

import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

teacherForcingRatio = 0.5

'''
    Input : 
        sourceTensor :      Tensor of indices for the source word
        targetTensor :      Tensor of indices for the target word
        encoder :           Encoder Object
        decoder :           Decoder Object
        encoderOptimizer :  Optimizer used for encoder
        decoderOptimizer :  Optimizer used for decoder
        criterion :         TODO
        maxLength :         TODO
'''

class Training :
    
    def __init__(self, dataProcessor) :
        self.dataProcessor = dataProcessor
        self.maxLength = dataProcessor.getMaxLength()

    def train(self, sourceTensor, targetTensor, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion) :

        # Tensor of zeros with shape(1, 1, hidden_size)
        encoderHidden = encoder.initHidden()

        # Make the gradients zero for encodeOptimizer and decoderOptimizer
        encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()

        # Length of Source Tensor
        sourceTensorLength = sourceTensor.size(0)

        # Length of Target Tensor
        targetTensorLength = targetTensor.size(0)

        # encoderOutputs = torch.zeros(self.maxLength, encoder.hidden_size, device = self.dataProcessor.device)

        # Initialize training loss to zero
        loss = 0

        # Traverse over the sourceTensor
        for sourceIndex in range(sourceTensorLength):

            # Pass the charIndex present at sourceTensor[sourceIndex] as input tensor
            encoderOutput, encoderHidden = encoder(sourceTensor[sourceIndex], encoderHidden)

            # Stores the encoder output for current source character in encoderOutputs [used in attention decoders]
            # encoderOutputs[sourceIndex] = encoderOutput[0, 0]

        # Give SOW (Start of Word) as Decoder Input
        decoderInput = torch.tensor([[self.dataProcessor.SOW2Int]], device = self.dataProcessor.device)

        # The hidden state input to decoder is the final hidden state output of encoder
        decoderHidden = encoderHidden

        # Randomly choose to use teacher forcing or not
        # useTeacherForcing = True if random.random() < teacherForcingRatio else False
        useTeacherForcing = False

        # Teacher forcing: Feed the target as the next input
        if useTeacherForcing:

            # Traverse over the target tensor
            for di in range(targetTensorLength) :

                # Pass the decoderInput, encoderOutputs, and decoderHidden to the decoder
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)

                # Calculate Loss between the index of character predicted and actual index of character
                loss += criterion(decoderOutput, targetTensor[di])

                # Teacher Forcing # Feed the target as the next input
                decoderInput = targetTensor[di]

        # Without teacher forcing: use its own predictions as the next input
        else:
            
            # Traverse over the target tensor
            for di in range(targetTensorLength) :

                # Pass the decoderInput, encoderOutputs, and decoderHidden to the decoder
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)

                # Returns the top value and top index in decoderOutput -> decoderOutput is output from softmax # TODO ? Is the syntax correct??
                topv, topi = decoderOutput.topk(1)

                # Squeeze to remove all dimensions that are 1
                # Detach returns a new tensor detached from the current history graph
                decoderInput = topi.squeeze().detach()

                # Calculate Loss between the index of character predicted and actual index of character
                loss += criterion(decoderOutput, targetTensor[di])
                
                # If decoder outputs EOW, it has generated an entire word
                if decoderInput.item() == self.dataProcessor.EOW2Int:
                    break

        # Once the forward pass has been done, start the backward pass through time
        loss.backward()

        # Back Propagate through time
        encoderOptimizer.step()
        decoderOptimizer.step()

        # Return average loss for each character in target word
        return loss.item() / targetTensorLength

    def trainIters(self, encoder, decoder, nIters, printEvery = 1000, plotEvery = 100, learningRate = 0.01):

        plot_losses = []
        print_loss_total = 0  # Reset every printEvery
        plot_loss_total = 0  # Reset every plotEvery

        # We will use SGD for both encoder and decoder 
        encoder_optimizer = optim.SGD(encoder.parameters(), lr = learningRate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr = learningRate)

        # List of tuple of train source words and train target words
        trainPairs = self.dataProcessor.createTrainPairs()

        # List of tuple of train source word converted to tensor and train target word converted to tensor
        trainingTensorPairs = []
        for i in range(nIters) :
            trainingTensorPairs.append(self.dataProcessor.tensorsFromPair(trainPairs[i]))
            
        # Loss Function
        criterion = nn.NLLLoss()

        for iter in range(1, nIters + 1) :

            # Select a pair from training data
            trainingPair = trainingTensorPairs[iter - 1]
            sourceTensor = trainingPair[0]
            targetTensor = trainingPair[1]

            # Train the module for train data pair
            loss = self.train(sourceTensor, targetTensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            # Store the loss for printing and plotting
            print_loss_total += loss
            plot_loss_total += loss

            if iter % printEvery == 0:

                # Calculate average loss
                print_loss_avg = print_loss_total / printEvery
                print_loss_total = 0
                print("Train Iter", iter, "/", nIters, " Loss : ", print_loss_avg )

            if iter % plotEvery == 0:

                # Calculate average loss
                plot_loss_avg = plot_loss_total / plotEvery

                # Append to list for plotting
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            
            # print('Training of Word Pair -', iter, "Completed.")

        self.evaluateSplit("Validation", encoder, decoder, criterion)
        self.evaluateSplit("Test", encoder, decoder, criterion)

        # Plot the list 
        self.showPlot(plot_losses)

    def showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()

        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        # plt.show()

    def evaluateWord(self, encoder, decoder, sourceWord, targetWord, criterion) :

        '''
        Input :
                encoder : Encoder Object
                decoder : Decoder Object
                word : Input Word to be transliterated
                maxLength : TODO
        '''

        with torch.no_grad():

            # Create tensor for source word
            sourceTensor = self.dataProcessor.tensorFromWord(self.dataProcessor.sourceCharToInt, sourceWord, "source")

            # Create tensor for target word
            targetTensor = self.dataProcessor.tensorFromWord(self.dataProcessor.targetCharToInt, targetWord, "target")

            # Length of Source Tensor
            sourceTensorLength = sourceTensor.size()[0]

            # Length of target Tensor
            targetTensorLength = targetTensor.size()[0]

            # Initialize the first hidden state for encoder
            encoderHidden = encoder.initHidden()

            # encoderOutputs = torch.zeros(self.maxLength, encoder.hidden_size, device = self.dataProcessor.device)

            # Iterate over the sourceTensor
            for sourceIndex in range(sourceTensorLength) :
                
                # Pass the charIndex present at sourceTensor[sourceIndex] as input tensor
                encoderOutput, encoderHidden = encoder(sourceTensor[sourceIndex], encoderHidden)
                
                # Stores the encoder output for current source character in encoderOutputs # TODO ? Why encoderOutput[0, 0]
                # encoderOutputs[sourceIndex] += encoderOutput[0, 0]

            # Give SOW (Start of Word) as Decoder Input
            decoderInput = torch.tensor([[self.dataProcessor.SOW2Int]], device = self.dataProcessor.device)  
            
            # The hidden state input to decoder is the final hidden state output of encoder
            decoderHidden = encoderHidden

            # List of decoded characters
            decodedCharacters = []

            # TODO -> Understand Attention
            # decoderAttentions = torch.zeros(maxLength, maxLength)

            loss = 0

            # Iterate till we get EOW
            for di in range(self.maxLength):

                # Pass the decoderInput, decoderHidden and encoderOutputs to the decoder
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                
                # decoderAttentions[di] = decoder_attention.data
                
                # Since the output of decoder is softmax, we want the value and index of maximum probability
                topv, topi = decoderOutput.data.topk(1)
                
                # EOW
                if topi.item() == self.dataProcessor.EOW2Int:
                    break
                
                else:

                    # Add the unique index of character in target language
                    decodedCharacters.append(self.dataProcessor.targetIntToChar[topi.item()])

                if di < targetTensorLength : 
                    loss += criterion(decoderOutput, targetTensor[di])

                # Squeeze removes all dimensions that are 1 from tensor
                # Detach creates a new tensor that is not the part of history graph
                decoderInput = topi.squeeze().detach()

            # return decodedCharacters, decoderAttentions[:di + 1]
            return loss.item() / targetTensorLength
        
    def evaluateSplit(self, split, encoder, decoder, criterion) :
        
        # List of tuple of source words and train target words
        if split.lower() == "validation" :
            pairs = self.dataProcessor.createValidPairs()

        else :
            pairs = self.dataProcessor.createTestPairs()

        # Initialize Total Loss for Validation Data
        totalLoss = 0

        # Accumulate Loss for all Words
        for i in range(len(pairs)) :
            totalLoss += self.evaluateWord(encoder, decoder, pairs[i][0], pairs[i][1], criterion)

            if i % (len(pairs) // 10) == 0 :
                print(split, "Data Evaluated :", i, "/", len(pairs))

        # Calculate Average Valid Loss
        avgValidLoss = totalLoss / len(pairs)

        print(split, "Loss :", avgValidLoss)
        return avgValidLoss


        