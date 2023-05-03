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

teacherForcingRatio = 0.5

class Training :
    
    def __init__(self, dataProcessor, encoder, decoder) :

        self.dataProcessor = dataProcessor # Store the dataProcessor object for using its member variables and functions
        
        self.encoder = encoder
        self.decoder = decoder

        # Store as class variables for readability
        self.maxLengthWord = dataProcessor.maxLengthWord
        self.maxLengthTensor = self.maxLengthWord + 1
        self.attention = dataProcessor.attention
        self.batchSize = self.dataProcessor.batchSize
        self.epochs = self.dataProcessor.epochs
        
    def trainOneBatch(self, sourceTensor, targetTensor, encoderOptimizer, decoderOptimizer, criterion) :

        ''' Train model for one batch and return batch loss. '''

        loss = 0

        sourceTensor = sourceTensor.squeeze()   # Squeeze to remove all dimensions = 1
        targetTensor = targetTensor.squeeze()

        # Create initial encoder hidden object
        encoderHidden = self.encoder.initHidden()

        if self.dataProcessor.cellType == 'LSTM' :
            encoderCell = self.encoder.initCell()
            encoderHidden = (encoderHidden, encoderCell)

        encoderOptimizer.zero_grad()    # Set gradients to zero before training 
        decoderOptimizer.zero_grad()

        sourceTensorLength = sourceTensor.size(0)   # Compute length of tensors for looping
        targetTensorLength = targetTensor.size(0)

        if self.attention :
            encoderOutputs = torch.zeros(self.maxLengthTensor, self.batchSize, self.encoder.hiddenSize, device = self.dataProcessor.device) # Store encoder outputs for attention

        for sourceIndex in range(sourceTensorLength): # Traverse over the sourceTensor

            encoderOutput, encoderHidden = self.encoder(sourceTensor[sourceIndex], encoderHidden) # Encode charIndex at sourceTensor[sourceIndex]
            if self.attention :
                encoderOutputs[sourceIndex] = encoderOutput[0] # Stores the encoder output for current source character in encoderOutputs

        decoderInput = torch.tensor([[self.dataProcessor.SOW2Int] * self.batchSize], device = self.dataProcessor.device) # Give SOW  as Decoder Input
        decoderHidden = encoderHidden # The hidden state input to decoder is the final hidden state output of encoder

        useTeacherForcing = True if random.random() < teacherForcingRatio else False # Randomly choose to use teacher forcing or not
        
        if useTeacherForcing: # Teacher forcing: Feed the target as the next input
            for di in range(targetTensorLength) : # Traverse over the target tensor 

                if self.attention :
                    decoderOutput, decoderHidden, decoderAttention = self.decoder(decoderInput, decoderHidden, encoderOutputs.reshape(self.batchSize, self.maxLengthTensor, self.encoder.hiddenSize)) 
                else :
                    decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden) 

                loss += criterion(decoderOutput, targetTensor[di]) # Calculate Loss between the index of character predicted and actual index of character
                decoderInput = targetTensor[di] # Teacher Forcing # Feed the target as the next input
                
        else : # Without teacher forcing: use its own predictions as the next input
            for di in range(targetTensorLength) : # Traverse over the target tensor
            
                if self.attention :
                    decoderOutput, decoderHidden, decoderAttention = self.decoder(decoderInput, decoderHidden, encoderOutputs.reshape(self.batchSize, self.maxLengthTensor, self.encoder.hiddenSize))

                else :
                    decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden)

                topv, topi = decoderOutput.topk(1) # Returns the top value and top index in decoderOutput -> decoderOutput is output from softmax 
                decoderInput = topi.squeeze().detach() # Squeeze to remove all dimensions that are 1 # Detach returns a new tensor detached from the current history graph
                
                loss += criterion(decoderOutput, targetTensor[di]) # Calculate Loss between the index of character predicted and actual index of character
            
        loss.backward() # Once the forward pass has been done, start the backward pass through time
        
        # Back Propagate through time
        encoderOptimizer.step()
        decoderOptimizer.step()

        return loss.item() / targetTensorLength # Return average loss for each character in target word

    def train(self):

        encoder_optimizer = optim.NAdam(self.encoder.parameters(), lr = self.dataProcessor.learningRate)
        decoder_optimizer = optim.NAdam(self.decoder.parameters(), lr = self.dataProcessor.learningRate)
    
        criterion = nn.NLLLoss() # Loss Function
        
        # Loaders for Loading Train and Valid Data
        trainLoader = DataLoader(self.dataProcessor.trainingTensorPairs, batch_size = self.batchSize, shuffle = True)
        validLoader = DataLoader(self.dataProcessor.validTensorPairs, batch_size = self.batchSize, shuffle = True)
        
        for epoch in range(self.epochs) :

            epochLoss = 0 
            batchNumber = 0
            totalBatches = self.dataProcessor.numTrainPairs // self.batchSize

            for sourceTensor, targetTensor in trainLoader :

                sourceTensor = sourceTensor.transpose(0, 1)
                targetTensor = targetTensor.transpose(0, 1)

                batchLoss = self.trainOneBatch(sourceTensor, targetTensor, encoder_optimizer, decoder_optimizer, criterion) # Train each batch
                
                epochLoss += batchLoss # Accumulate Loss
                batchNumber += 1

                if (batchNumber % (totalBatches // 10)) == 0 :
                    print('Loss For Batch ' + str(batchNumber) + '/' + str(totalBatches) + ' : ' + str(batchLoss))

            epochLoss /= totalBatches # Average Epoch Loss
            validationLoss, validationAccuracy = self.evaluate(validLoader)

            print('Loss For Epoch ' + str(epoch + 1) + '/' + str(self.epochs) + ' : ' + str(epochLoss) + '\n')
            print('Validation Loss :', validationLoss)
            print('Validation Accuracy :', validationAccuracy, '%')

            wandb.log({'train loss':epochLoss,'validation loss':validationLoss, 'validation accuracy':validationAccuracy})
    
    def evaluate(self, loader) :
        
        ''' Compute loss and accuracy for loader. '''

        loss = 0
        totalCorrectWords = 0
        
        batchNumber = 1
        
        totalWords = len(loader.sampler)
        totalBatches = len(loader.sampler) // self.dataProcessor.batchSize

        criterion = nn.NLLLoss() # Loss Function

        for sourceTensor, targetTensor in loader :
            batchLoss, correctWords = self.evaluateOneBatch(sourceTensor, targetTensor, criterion) # Evaluate each batch

            loss += batchLoss
            totalCorrectWords += correctWords

            if batchNumber % (totalBatches // 10) == 0 :
                print("Evaluate Batch : " + str(batchNumber) + "/" + str(totalBatches))
            
            batchNumber += 1 

        return (loss / totalBatches), (100 * totalCorrectWords / totalWords)

    def evaluateOneBatch(self, sourceTensorBatch, targetTensorBatch, criterion) :

        ''' Evaluate a batch and return loss and number of correct words. '''

        loss = 0
        correctWords = 0

        batchSize = self.dataProcessor.batchSize
        device = self.dataProcessor.device
        maxLengthWord = self.dataProcessor.maxLengthWord 
        
        sourceTensor = Variable(sourceTensorBatch.transpose(0, 1))
        targetTensor = Variable(targetTensorBatch.transpose(0, 1))
        
        sourceTensorLength = sourceTensor.size()[0]
        targetTensorLength = targetTensor.size()[0]

        predictedBatchOutput = torch.zeros(targetTensorLength, batchSize, device = self.dataProcessor.device)

        encoderHidden = self.encoder.initHidden() # Initialize initial hidden state of encoder
        if self.dataProcessor.cellType == "LSTM":
            encoderCell = self.encoder.initHidden()
            encoderHidden = (encoderHidden, encoderCell)

        if self.attention:
            encoderOutputs = torch.zeros(maxLengthWord + 1, batchSize, self.encoder.hiddenSize, device = device) # Store encoder outputs for attention

        for ei in range(sourceTensorLength):
            encoderOutput, encoderHidden = self.encoder(sourceTensor[ei], encoderHidden) # Encode each charIndex

            if self.attention :
                encoderOutputs[ei] = encoderOutput[0]

        decoderInput = torch.tensor([[self.dataProcessor.SOW2Int] * batchSize], device = device) # Initialize input to decoder with start of word token
        decoderHidden = encoderHidden # initial hidden state for decoder will be final hidden state of encoder
        
        if self.attention :
            decoderAttentions = torch.zeros(self.maxLengthTensor, self.maxLengthTensor) # Store attentions for plotting
        
        for di in range(targetTensorLength):

            # Compute decoder outputs
            if self.attention :
                decoderOutput, decoderHidden, decoderAttention = self.decoder(decoderInput, decoderHidden, encoderOutputs.reshape(self.batchSize, self.maxLengthTensor, self.encoder.hiddenSize))
                decoderAttentions[di] = decoderAttention.data

            else : 
                decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden)
            
            loss += criterion(decoderOutput, targetTensor[di].squeeze()) # Compute point loss

            # Store predicted outputs
            topv, topi = decoderOutput.data.topk(1)
            decoderInput = torch.cat(tuple(topi))
            predictedBatchOutput[di] = torch.cat(tuple(topi))

        if self.attention :
                # attention
                plt.matshow(decoderAttentions[:di + 1].numpy())

        predictedBatchOutput = predictedBatchOutput.transpose(0,1) # Transpose predicted output for row to row comparison

        ignore = [self.dataProcessor.SOW2Int, self.dataProcessor.EOW2Int, self.dataProcessor.PAD2Int] 
        
        for di in range(predictedBatchOutput.size()[0]):

            predicted = [letter.item() for letter in predictedBatchOutput[di] if letter not in ignore]
            actual = [letter.item() for letter in targetTensorBatch[di] if letter not in ignore]

            if predicted == actual:
                correctWords += 1
        
        return loss.item(), correctWords