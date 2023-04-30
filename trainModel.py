from __future__ import unicode_literals, print_function, division
import torch
import random
import wandb

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

        encoderHidden = encoder.initHidden()

        if self.dataProcessor.cellType == 'LSTM' :
            encoderCell = encoder.initCell()
            encoderHidden = (encoderHidden, encoderCell)

        encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()

        sourceTensorLength = sourceTensor.size(0)
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
        useTeacherForcing = True if random.random() < teacherForcingRatio else False

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
        
        for epoch in range(self.epochs) :

            epochLoss = 0 
            batchNumber = 0
            totalBatches = self.dataProcessor.numTrainPairs // self.batchSize

            for sourceTensor, targetTensor in trainLoader :

                sourceTensor = sourceTensor.transpose(0, 1)
                targetTensor = targetTensor.transpose(0, 1)

                batchLoss = self.trainOneBatch(sourceTensor, targetTensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                epochLoss += batchLoss
                batchNumber += 1

                if (batchNumber % (totalBatches // 10)) == 0 :
                    print('Loss For Batch ' + str(batchNumber) + '/' + str(totalBatches) + ' : ' + str(batchLoss))

            epochLoss /= totalBatches
            # validationLoss, validationAccuracy = self.evaluate(validLoader, encoder, decoder)
            validationLoss, validationAccuracy = self.evaluate(validLoader, encoder, decoder)

            print('Loss For Epoch ' + str(epoch + 1) + '/' + str(self.epochs) + ' : ' + str(epochLoss) + '\n')
            print('Validation Loss :', validationLoss)
            print('Validation Accuracy :', validationAccuracy, '%')

            # wandb.log({'train loss':epochLoss,'validation loss':validationLoss, 'validation accuracy':validationAccuracy})
    
    def evaluate(self, loader, encoder, decoder) :

        loss = 0
        totalCorrectWords = 0
        batchNumber = 1
        
        totalWords = len(loader.sampler)
        totalBatches = len(loader.sampler) // self.dataProcessor.batchSize

        # Loss Function
        criterion = nn.NLLLoss()

        for sourceTensor, targetTensor in loader :
            batchLoss, correctWords = self.evaluateOneBatch(sourceTensor, targetTensor, encoder, decoder, criterion)

            loss += batchLoss
            totalCorrectWords += correctWords

            if batchNumber % (totalBatches // 10) == 0 :
                print("Evaluate Batch : " + str(batchNumber) + "/" + str(totalBatches))
            
            batchNumber += 1 

        return (loss / totalBatches), (100 * totalCorrectWords / totalWords)

    def evaluateOneBatch(self, sourceTensorBatch, targetTensorBatch, encoder, decoder, criterion) :

        loss = 0
        correctWords = 0

        batchSize = self.dataProcessor.batchSize
        device = self.dataProcessor.device
        maxLengthWord = self.dataProcessor.maxLengthWord 
        
        sourceTensor = Variable(sourceTensorBatch.transpose(0, 1))
        targetTensor = Variable(targetTensorBatch.transpose(0, 1))
        
        # Get source length
        sourceTensorLength = sourceTensor.size()[0]
        targetTensorLength = targetTensor.size()[0]

        predictedBatchOutput = torch.zeros(targetTensorLength, batchSize, device = self.dataProcessor.device)

        # Initialize initial hidden state of encoder
        encoderHidden = encoder.initHidden()

        if self.dataProcessor.cellType == "LSTM":
            encoderCell = encoder.initHidden()
            encoderHidden = (encoderHidden, encoderCell)

        if self.attention:
            encoderOutputs = torch.zeros(maxLengthWord + 1, batchSize, encoder.hiddenSize, device = device)

        for ei in range(sourceTensorLength):
            encoderOutput, encoderHidden = encoder(sourceTensor[ei], encoderHidden)

            if self.attention :
                # encoderOutputs[sourceIndex] = encoderOutput[0] # ? Other Code
                encoderOutputs[ei] = encoderOutput[0]

        # Initialize input to decoder with start of word token
        decoderInput = torch.tensor([[self.dataProcessor.SOW2Int] * batchSize], device = device)

        # initial hidden state for decoder will be final hidden state of encoder
        decoderHidden = encoderHidden

        if self.attention :
            decoderAttentions = torch.zeros(self.maxLengthTensor, self.maxLengthTensor)
        
        for di in range(targetTensorLength):

            if self.attention :
                decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)
                decoderAttentions[di] = decoderAttention.data

            else : 
                # Pass the decoderInput, decoderHidden and encoderOutputs to the decoder
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
            
            loss += criterion(decoderOutput, targetTensor[di].squeeze())

            topv, topi = decoderOutput.data.topk(1)
            decoderInput = torch.cat(tuple(topi))
            predictedBatchOutput[di] = torch.cat(tuple(topi))

        if self.attention :
            if False :
                plt.matshow(decoderAttentions[:di + 1].numpy())

        predictedBatchOutput = predictedBatchOutput.transpose(0,1)

        ignore = [self.dataProcessor.SOW2Int, self.dataProcessor.EOW2Int, self.dataProcessor.PAD2Int]

        for di in range(predictedBatchOutput.size()[0]):

            predicted = [letter.item() for letter in predictedBatchOutput[di] if letter not in ignore]
            actual = [letter.item() for letter in targetTensorBatch[di] if letter not in ignore]

            if predicted == actual:
                correctWords += 1
        
        return loss.item(), correctWords