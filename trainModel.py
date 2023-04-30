from __future__ import unicode_literals, print_function, division
import torch
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

teacherForcingRatio = 0.5

class Training :
    
    def __init__(self, dataProcessor) :

        # Store the dataProcessor object for using its member variables and functions
        self.dataProcessor = dataProcessor

        # Store the maxLength in class variable
        self.maxLengthWord = dataProcessor.maxLengthWord
        self.maxLengthTensor = self.maxLengthWord + 1

        self.attention = dataProcessor.attention

        self.batchSize = self.dataProcessor.batchSize
        self.epochs = self.dataProcessor.epochs

    def trainOneBatch(self, sourceTensor, targetTensor, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion) :

        sourceTensor = sourceTensor.squeeze()
        targetTensor = targetTensor.squeeze()

        # Tensor of zeros with shape(1, 1, hidden_size)
        encoderHidden = encoder.initHidden()

        if self.dataProcessor.cellType == 'LSTM' :
            encoderCell = encoder.initCell()
            encoderHidden = (encoderHidden, encoderCell)

        # Make the gradients zero for encodeOptimizer and decoderOptimizer
        encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()

        # Length of Source Tensor
        sourceTensorLength = sourceTensor.size(0)

        # Length of Target Tensor
        targetTensorLength = targetTensor.size(0)

        if self.attention :
            encoderOutputs = torch.zeros(self.maxLengthTensor, self.batchSize, encoder.hiddenSize, device = self.dataProcessor.device)

        # Initialize training loss to zero
        loss = 0

        # Traverse over the sourceTensor
        for sourceIndex in range(sourceTensorLength):

            # Pass the charIndex present at sourceTensor[sourceIndex] as input tensor
            encoderOutput, encoderHidden = encoder(sourceTensor[sourceIndex], encoderHidden)

            if self.attention :
                # Stores the encoder output for current source character in encoderOutputs [used in attention decoders]
                encoderOutputs[sourceIndex] = encoderOutput[0]

        # Give SOW (Start of Word) as Decoder Input
        decoderInput = torch.tensor([[self.dataProcessor.SOW2Int] * self.batchSize], device = self.dataProcessor.device)

        # The hidden state input to decoder is the final hidden state output of encoder
        decoderHidden = encoderHidden

        # Randomly choose to use teacher forcing or not
        # useTeacherForcing = True if random.random() < teacherForcingRatio else False
        useTeacherForcing = False

        # Teacher forcing: Feed the target as the next input
        if useTeacherForcing:

            # Traverse over the target tensor
            for di in range(targetTensorLength) :

                if self.attention :
                    # Pass the decoderInput, and decoderHidden to the decoder
                    decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)

                else :

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

                if self.attention :
                    decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)

                else :
                    # Pass the decoderInput, encoderOutputs, and decoderHidden to the decoder
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)

                # Returns the top value and top index in decoderOutput -> decoderOutput is output from softmax 
                topv, topi = decoderOutput.topk(1)

                # Squeeze to remove all dimensions that are 1
                # Detach returns a new tensor detached from the current history graph
                decoderInput = topi.squeeze().detach()

                # Calculate Loss between the index of character predicted and actual index of character
                loss += criterion(decoderOutput, targetTensor[di])
                
                # If decoder outputs EOW, it has generated an entire word
                # if decoderInput.item() == self.dataProcessor.EOW2Int:
                #     break

        # Once the forward pass has been done, start the backward pass through time
        loss.backward()

        # Back Propagate through time
        encoderOptimizer.step()
        decoderOptimizer.step()

        # Return average loss for each character in target word
        return loss.item() / targetTensorLength

    def train(self, encoder, decoder, learningRate = 0.001):

        # We will use SGD for both encoder and decoder 
        encoder_optimizer = optim.NAdam(encoder.parameters(), lr = self.dataProcessor.learningRate)
        decoder_optimizer = optim.NAdam(decoder.parameters(), lr = self.dataProcessor.learningRate)
    
        # Loss Function
        criterion = nn.NLLLoss()
    
        trainLoader = DataLoader(self.dataProcessor.trainingTensorPairs, batch_size = self.batchSize, shuffle = True)
        validLoader = DataLoader(self.dataProcessor.validTensorPairs, batch_size = self.batchSize, shuffle = True)
        
        plot_losses = []
        plot_loss_total = 0  # Reset every plotEvery

        for epoch in range(self.epochs) :

            epochLoss = 0 
            batchNumber = 0
            totalBatches = self.dataProcessor.numTrainPairs // self.batchSize

            for sourceTensor, targetTensor in trainLoader :

                sourceTensor = sourceTensor.transpose(0, 1)
                targetTensor = targetTensor.transpose(0, 1)

                batchLoss = self.trainOneBatch(sourceTensor, targetTensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

                # Store the loss for printing and plotting
                epochLoss += batchLoss
                plot_loss_total += batchLoss

                batchNumber += 1

                if (batchNumber % (totalBatches // 10)) == 0 :
                    print('Loss For Batch ' + str(batchNumber) + '/' + str(totalBatches) + ' : ' + str(batchLoss))

            epochLoss /= totalBatches

            print('Loss For Epoch ' + str(epoch + 1) + '/' + str(self.epochs) + ' : ' + str(epochLoss) + '\n')

            # print(' Train Accuracy :', self.evaluate(trainLoader, encoder, decoder))
            print(' Validation Accuracy :', self.evaluate(validLoader, encoder, decoder))

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

    def evaluate(self, loader, encoder, decoder) :
    
        batchSize = self.dataProcessor.batchSize
        device = self.dataProcessor.device
        maxLengthWord = self.dataProcessor.maxLengthWord 
        
        totalWords = 0
        correctWords = 0
        batchNumber = 1
        totalBatches = len(loader.sampler) // batchSize
            
        for sourceTensor, targetTensor in loader:
            
            source_tensor = Variable(sourceTensor.transpose(0, 1))
            target_tensor = Variable( targetTensor.transpose(0, 1))
            
            # Get source length
            source_length = source_tensor.size()[0]
            target_length = target_tensor.size()[0]
            output = torch.LongTensor(target_length, batchSize)

            # Initialize initial hidden state of encoder
            encoder_hidden = encoder.initHidden()

            if self.dataProcessor.cellType == "LSTM":
                encoder_cell_state = encoder.initHidden()
                encoder_hidden = (encoder_hidden,encoder_cell_state)
    
            if self.attention:
                encoder_outputs = torch.zeros(maxLengthWord + 1, batchSize, encoder.hiddenSize, device = device)

            for ei in range(source_length):
                encoder_output, encoder_hidden = encoder(source_tensor[ei],encoder_hidden)

                if self.attention :
                    # encoderOutputs[sourceIndex] = encoderOutput[0] # ? Other Code
                    encoder_outputs[ei] = encoder_output[0]

            # Initialize input to decoder with start of word token
            decoder_input = torch.tensor([[self.dataProcessor.SOW2Int] * batchSize], device = device)

            # initial hidden state for decoder will be final hidden state of encoder
            decoder_hidden = encoder_hidden

            if self.attention :
                decoderAttentions = torch.zeros(self.maxLengthTensor, self.maxLengthTensor)
            
            for di in range(target_length):

                if self.attention :
                    decoder_output, decoder_hidden, decoderAttention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoderAttentions[di] = decoderAttention.data

                else : 
                    # Pass the decoder_input, decoder_hidden and encoderOutputs to the decoder
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                topv, topi = decoder_output.data.topk(1)
                decoder_input = torch.cat(tuple(topi))
                output[di] = torch.cat(tuple(topi))

            if False :
                plt.matshow(decoderAttentions[:di + 1].numpy())

            output = output.transpose(0,1)
            
            for di in range(output.size()[0]):
                ignore = [self.dataProcessor.SOW2Int, self.dataProcessor.EOW2Int, self.dataProcessor.PAD2Int]
                predicted = [letter.item() for letter in output[di] if letter not in ignore]
                actual = [letter.item() for letter in  targetTensor[di] if letter not in ignore]
                # print("predicted:",predicted)
                # print("=actual:",actual)
                if predicted == actual:
                    correctWords += 1
                totalWords += 1
            print("Evaluate Batch : " + str(batchNumber) + "/" + str(totalBatches))
            batchNumber+=1
        # print('accuracy '+ str((correct/total)*100))
        return (correctWords/totalWords)*100


















    def evaluateBatch(self, sourceTensor, targetTensor, encoder, decoder, criterion) :

        '''
        Input :
                encoder : Encoder Object
                decoder : Decoder Object
                word : Input Word to be transliterated
                maxLength : TODO
        '''

        loss = 0
        correctWords = 0

        with torch.no_grad():

            sourceTensor = sourceTensor.squeeze()
            targetTensor = targetTensor.squeeze()

            # Length of Source Tensor
            sourceTensorLength = sourceTensor.size()[0]

            # Length of target Tensor
            targetTensorLength = targetTensor.size()[0]

            # Initialize the first hidden state for encoder
            encoderHidden = encoder.initHidden()

            if self.dataProcessor.cellType == 'LSTM' :
                encoderCell = encoder.initCell()
                encoderHidden = (encoderHidden, encoderCell)

            if self.attention :
                encoderOutputs = torch.zeros(self.maxLengthTensor, self.dataProcessor.batchSize, encoder.hiddenSize, device = self.dataProcessor.device)

            output = torch.LongTensor(targetTensorLength, self.dataProcessor.batchSize)

            # Iterate over the sourceTensor
            for sourceIndex in range(sourceTensorLength) :
                
                # Pass the charIndex present at sourceTensor[sourceIndex] as input tensor
                encoderOutput, encoderHidden = encoder(sourceTensor[sourceIndex], encoderHidden)

                if self.attention : 
                    
                    # Stores the encoder output for current source character in encoderOutputs
                    # encoderOutputs[sourceIndex] = encoderOutput[0] # ? Other Code
                    encoderOutputs[sourceIndex] = encoderOutput[0]

            # Give SOW (Start of Word) as Decoder Input
            decoderInput = torch.tensor([[self.dataProcessor.SOW2Int] * self.batchSize], device = self.dataProcessor.device)

            # The hidden state input to decoder is the final hidden state output of encoder
            decoderHidden = encoderHidden

            if self.attention :
                decoderAttentions = torch.zeros(self.maxLengthTensor, self.maxLengthTensor)

            for targetIndex in range(targetTensorLength):
                
                if self.attention :
                    decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)

                else : 
                    # Pass the decoderInput, decoderHidden and encoderOutputs to the decoder
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                
                if self.attention :
                    decoderAttentions[targetIndex] = decoderAttention.data
                
                # Since the output of decoder is softmax, we want the value and index of maximum probability
                topv, topi = decoderOutput.data.topk(1)
                loss += criterion(decoderOutput, targetTensor[targetIndex])

                # Squeeze removes all dimensions that are 1 from tensor
                # Detach creates a new tensor that is not the part of history graph
                decoderInput = topi.squeeze().detach()
                output[targetIndex] = decoderInput

            if False :
                plt.matshow(decoderAttentions[:di + 1].numpy())

            output = output.transpose(0, 1)
            print(output.shape)

            print("OUTPUT SIZE : ", output.size()[0])
            for batchNumber in range(output.size()[0]) :
                ignore = [self.dataProcessor.SOW2Int, self.dataProcessor.EOW2Int, self.dataProcessor.PAD2Int]
                predicted = [self.dataProcessor.targetIntToChar[letter.item()] for letter in output[batchNumber] if letter not in ignore]

                print("Length of target tensor : ", targetTensor[batchNumber].shape)
                actual = [self.dataProcessor.targetIntToChar[letter.item()] for letter in targetTensor[batchNumber] if letter not in ignore]

                if predicted == actual :
                    correctWords += 1
            
            # Return loss and correctWord = 1
            return loss.item() / targetTensorLength , correctWords
        
    def evaluateSplit(self, split, encoder, decoder, criterion) :
        
        # List of tuple of source words and train target words
        if split.lower() == "validation" :
            tensorPairs = self.dataProcessor.validTensorPairs

        else :
            tensorPairs = self.dataProcessor.testTensorPairs

        loader = DataLoader(tensorPairs, batch_size = self.batchSize, shuffle = True)

        # Initialize Total Loss for Validation Data
        totalLoss = 0

        # Initialize Number of Correct Words
        correctWords = 0

        batchNumber = 0
        totalBatches = len(tensorPairs) // self.batchSize

        for sourceTensor, targetTensor in loader :

            sourceTensor = sourceTensor.transpose(0, 1)
            targetTensor = targetTensor.transpose(0, 1)

            batchLoss, correctWordsBatch = self.evaluateBatch(sourceTensor, targetTensor, encoder, decoder, criterion)

            totalLoss += batchLoss
            correctWords += correctWordsBatch
            batchNumber += 1

        avgLoss = totalLoss / totalBatches
        accuracy = correctWords / len(tensorPairs)

        print(split, "Loss :", avgLoss)
        print(split, "Accuracy :", accuracy)

        return avgLoss, accuracy

