# DLAssignment3
CS6910 Deep Learning Assignment 3 - Building and training a RNN model from scratch for seq2seq character level Neural machine transliteration.

  The goal of this assignment is threefold: 
  
    (i) learn how to model sequence-to-sequence learning problems using Recurrent Neural Networks 
    (ii) compare different cells such as vanilla RNN, LSTM and GRU 
    (iii) understand how attention networks overcome the limitations of vanilla seq2seq models.

The *wandb report* can be found in the following link:

https://wandb.ai/cs22m033/DL%20Assignment%203/reports/CS6910-Assignment-3-Seq2Seq-Character-level-Neural-Machine-Transliteration--Vmlldzo0Mzk5Nzk4

## Dataset

We have used a sample of the Aksharantar dataset released by AI4Bharat. This dataset contains pairs of the following form: 

    ajanabee,अजनबी

We have used the *Hindi* dataset. The dataset contains word pairs where one word is in English, while the other word represents the English word in Devnagiri Script. 

## Usage
### To run code with wandb -

python train.py 
 
### To run code without wandb -

- *With Attention*
python main.py -a 1

- *Without Attention*
python main.py -a 0

Argparse is coded in the main.py file. Use the following parameters with the command "python main.py" in order to train the model for the various parameters.
 
## Using argparse
| Command Line Argument | Usage |
| --- | --- |
| --wandb_project / -wp  | Name of wandb project |
| --wandb_entity / -we  | Name of wandb entity |
| --hidden_size / -hs  | Hidden size of dense layer |
| --cell_type / -c  | Cell type to use - lstm,gru,rnn |
| --numLayersEncoder / -nle  | Number of encoder layers |
| --numLayersDecoder / -nld  | Number of decoder layers |
| --drop_out / -dp  | Dropout |
| --embedding_size / -es  | Embedding size |
| --batch_size / -bs  | Batch size |
| --epoch / -e  | Epochs |
| --learning_rate / -lr | Learning Rate |
| --attention / -a | Attention (1 = True, 0 = False) |
| --bidirectional / -bd | Bidirectional (1 = True, 0 = False) |

The following hyperparameters have been used for wandb sweeps (both with and without attention) :

| Hyperparameter | Values/Usage |
| --- | --- |
| epoch | 5, 10 |
| hidden_size | 128, 256, 512 |
| cell_type | LSTM, RNN, GRU |
| learning_rate | 1e-2, 1e-3 |
| num_layers | 1, 2, 3 |
| drop_out | 0, 0.2, 0.3 |
| embedding_size | 64, 128, 256, 512 |
| batch_size | 32, 64, 128 |
| bidirectional | True, False |

## Files

- dataPreprocessing.py - Preprocesses data and creates train,test and validation data splits. Also creates source and target language vocabularies.
- main.py - Driver File (with argparse) to train the models
- model.py - Contains the Encoder,Decoder and DecoderAttention classes, modelled according to the OOPs concepts.
- train.py - Driver File used to execute sweeps
- trainModel.py - Contains the code to train the model according to the configs and compute corresponding loss and accuracy

## Folders
- predictions_vanilla - Contains a csv file containing the input words, predictions made by the model (without attention) and target words
- predictions_attention - Contains a csv file containing the input words, predictions made by the model (with attention) and target words
- heatMaps - Contains heatMaps for few of the predictions

## Results

### With Attention

####  Validation Accuracy - 34.985 %
####  Test Accuracy - 33.569 %
| Hyperparameter | Values/Usage |
| --- | --- |
| epoch | 15 |
| hidden_size | 256 |
| cell_type | LSTM |
| learning_rate | 1e-3 |
| num_layersEncoder | 3 |
| num_layersDecoder | 3 |
| drop_out | 0 |
| embedding_size | 256 |
| batch_size | 64 |
| bidirectional | False |

### Without Attention

####  Validation Accuracy - 34.644 %
#### Test Accuracy - 31.86 %
| Hyperparameter | Values/Usage |
| --- | --- |
| epoch | 10 |
| hidden_size | 256 |
| cell_type | LSTM |
| learning_rate | 1e-3 |
| num_layersEncoder | 2 |
| num_layersDecoder | 2 |
| drop_out | 0.2 |
| embedding_size | 256 |
| batch_size | 64 |
| bidirectional | False |
