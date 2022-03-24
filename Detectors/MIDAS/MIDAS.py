import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pytorch_transformers import *
import time
from torch.utils.data import Dataset, DataLoader
import copy
from random import randrange
from torch.optim import lr_scheduler
import os
import csv
import emoji
import wordsegment
import sys
import nltk
from nltk.corpus import stopwords
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import string
wordsegment.load()


# a MIDAS class object which allows single query tests, assumes model was pretrained and the model is stored
class MIDAS:

    def __init__(self, train_data = 'OFFis1_simpleDebiased_offenseval-training-v1.tsv', trained_cnn_model = 'MIDAS_CNN.pt', trained_blstm_model = 'MIDAS_BLSTM.pt', trained_blstmGru_model = 'MIDAS_BLSTM-GRU.pt'):

        self.tokenize = lambda x: nltk.word_tokenize(x.lower())
        
        self.TEXT = Field(sequential = True, tokenize = self.tokenize, lower = True, include_lengths=True)
        self.LABEL = Field(sequential = False, use_vocab = False, dtype = torch.float)
        self.ID = Field(sequential = False, use_vocab = False)
        
        off_datafields = [('id', None), ('text', self.TEXT), ('label', self.LABEL), ('is_target', None), ('target', None)]
        
        trn = TabularDataset.splits(path='.', train=train_data, format='tsv', fields=off_datafields)[0]

        self.TEXT.build_vocab(trn, vectors='glove.6B.200d')
        
        self.BATCH_SIZE = 64
                
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        # load pre-trained model
        if torch.cuda.is_available():                                                                                                                                      
            print ("Using cuda")                                                                                                                                           
            self.cnn_model = torch.load(trained_cnn_model)
            self.blstm_model = torch.load(trained_blstm_model)
            self.blstmGru_model = torch.load(trained_blstmGru_model)
        else:                                                                                                                                                              
            print("Using cpu")                                           
            self.cnn_model = torch.load(trained_cnn_model, map_location = 'cpu')
            self.blstm_model = torch.load(trained_blstm_model, map_location = 'cpu')
            self.blstmGru_model = torch.load(trained_blstmGru_model, map_location = 'cpu')

  


    def predict(self, test_query):


        '''
        tmpFile = open('tmpTstFile.tsv', 'w')
        tmpFile.write('1' + '\t' + test_query.replace('\0', ''))
        tmpFile.close()
        
        tst_datafields = [('id', self.ID), ('text', self.TEXT)]

        tst = TabularDataset(path='tmpTstFile.tsv', format='tsv', fields = tst_datafields)

        test_iterator = Iterator(tst, batch_size=self.BATCH_SIZE, sort_key = lambda x: len(x.text), sort_within_batch=True)
        '''

        #load three models, get predictions, let vote on final prediction
        cnn_votes = {}
        
        #NOTE: These try-except blocks are here to deal with cases of text consisting only of stop words which cause a runtime error
        '''
        try:
            cnn_predictions, cnn_probs = test_cnn(self.cnn_model, test_iterator)
        
            for id, pred in cnn_probs:
                cnn_votes[str(id)] = pred

        except:
            cnn_votes['1'] = 0
        '''
        _, cnn_prob = cnn_predict_sentence(self.cnn_model, self.TEXT, self.device, test_query)
        cnn_votes['1'] = cnn_prob

        #BLSTM
        
        blstm_votes = {}
        
        '''
        try:
            blstm_predictions, blstm_probs = test_blstm(self.blstm_model, test_iterator)
            
            for id, pred in blstm_probs:
                blstm_votes[str(id)] = pred
        except:
            blstm_votes['1'] = 0
        '''
        _, blstm_prob = blstm_predict_sentence(self.blstm_model, self.TEXT, self.device, test_query)
        blstm_votes['1'] = blstm_prob

        #BLSTM_BGRU
        
        blstm_gru_votes = {}
        
        '''
        try:
            blstm_gru_predictions, blstm_gru_probs = test_blstm(self.blstmGru_model, test_iterator)
        
            for id, pred in blstm_gru_probs:
                blstm_gru_votes[str(id)] = pred
        except:
            blstm_gru_votes['1'] = 0
        '''
        _, blstm_gru_prob = blstm_predict_sentence(self.blstm_model, self.TEXT, self.device, test_query)
        blstm_gru_votes['1'] = blstm_gru_prob

        predictions = []
        probs = []

        # have each system vote for prediction (0 or 1) if at least 2 vote for 1 score will be 2 or higher, if at least 2 vote for 0 score will be less than 2
        for id in cnn_votes:
            #votes = cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id]
            #if(votes >= 2):
            #    pred = 1
            #else:
            #    pred = 0
            pred = int(round((cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id])/3))
            prob = (cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id])/3


            # work around for now as MIDAS returns 0 or 1 and needs to return OFF or NOT
            if(pred == 0):
                pred = 'NOT'
            else:
                pred = 'OFF'
            
            predictions.append(pred)
            probs.append(prob)



        return predictions[0], probs[0]



    # allows multiple queries to be tested at once.
    def predictMultiple(self, test_queries):
        
        predictions = []
        probs = []

        for test in test_queries:
            pred, prob = self.predict(test)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs







# CNN portion of MIDAS
# Example from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
# Helpful torchtext explanation: https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 256)
        self.out = nn.Linear(256, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        fc1 = self.dropout(self.fc(cat))
        
        # output
        
        return self.out(fc1)


# BLSTM portion of MIDAS
# Example from https://www.kaggle.com/maxl28618/toxic-comments-lstm-in-pytorch-with-torchtext
# Further expansion with relevant iterator: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
class BLSTM(nn.Module):


    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
 

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden.squeeze(0))


class BLSTM_GRU(nn.Module):


    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           bidirectional=True)

        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional = True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
 

    def forward(self, text, text_lengths):
        
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        # pass to BGRU
        packed_lstm_outputs = nn.utils.rnn.pack_padded_sequence(output, output_lengths)
        packed_output, (hidden, cell) = self.gru(packed_lstm_outputs)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        return self.fc(output[-1,:,:])


        
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train_cnn(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, _ = batch.text

        predictions = model(text).squeeze(1)
        
        #print(predictions)
        #print(batch.label)
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_blstm(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)
        
        
        #print(predictions)
        #print(batch.label)
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def cnn_predict_sentence(model, TEXT, device, sentence):
    model.eval()
    tokenized = [x for x in nltk.word_tokenize(sentence.lower())]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    try:
        prediction = torch.sigmoid(model(tensor))
        return round(prediction.item()), prediction.item()
    except:
        return 0, 0

def blstm_predict_sentence(model, TEXT, device, sentence):
    model.eval()
    tokenized = [x for x in nltk.word_tokenize(sentence.lower())]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    try:
        prediction = torch.sigmoid(model(tensor, length_tensor))
        return round(prediction.item()), prediction.item()
    except:
        return 0, 0
    
def test_cnn(model, iterator):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    final_preds = []
    pred_probs = []

    with torch.no_grad():
    
        for batch in iterator:
            text, _ = batch.text

            predictions = model(text).squeeze(1)
            
            #print(predictions)
            sigmoid_preds = torch.sigmoid(predictions)
            rounded_preds = torch.round(sigmoid_preds)
            

            # save each prediction with corresponding id to list to write to file later
            for i in range(len(batch.id)):
                final_preds.append((batch.id[i].item(), int(rounded_preds[i].item())))
                pred_probs.append((batch.id[i].item(), sigmoid_preds[i].item()))

    return final_preds, pred_probs

def test_blstm(model, iterator):
    
    model.eval()
    
    final_preds = []
    pred_probs = []

    with torch.no_grad():
    
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths)#.squeeze(1)
            
            #print(predictions)
            sigmoid_preds = torch.sigmoid(predictions)
            rounded_preds = torch.round(sigmoid_preds)
            
            # save each prediction with corresponding id to list to write to file later
            for i in range(len(batch.id)):
                final_preds.append((batch.id[i].item(), int(rounded_preds[i].item())))
                pred_probs.append((batch.id[i].item(), sigmoid_preds[i].item()))
            
    return final_preds, pred_probs



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# removes any non standard english characters
def convertChars(text):
    char_file = open('NamesList.txt')
    char_dict = {}
    
    # set up dictionary for unrecognizable characters
    for line in char_file:
        line = line.split('\t')
        
        if(len(line) == 1 or line[0] == '' or '@' in line[0]):
            continue
        else:
            char_dict[line[0]] = line[1].strip()

        
    standChars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # step through each character in text and replace if non standard
    out_text = ''
    
    for char in text:
        if char not in string.punctuation and char not in standChars and char != ' ':
            code = hex(ord(char))
            code = code[2:].upper() # remove '0x' from hex num
            code = '0' * (4 - len(code)) + code # pad to 4 length to match dictionary

            if(code in char_dict):
                desc = char_dict[code].lower().split()
                
                if('letter' in desc):
                    # special cases where character does not follow letter or is not standard char (e.g. 'letter eth')
                    if(len(desc[desc.index('letter')+ 1]) > 1):
                        if(len(desc) > desc.index('letter')+ 2):
                            out_char = desc[desc.index('letter') + 2][0]
                        else:
                            out_char = desc[desc.index('letter') + 1][0]
                    else:
                        # out char should be the character which follows letter
                        out_char = desc[desc.index('letter') + 1]

                else:
                    out_char = char
                    
            else:
                out_char = char
        else:
            out_char = char
            
        out_text += out_char
    return out_text
                    
                
                
            
    
# produces LM alignment scores for the testFile
def getLMPreds(testFile, LM):
    lm = CommunityLM(community = LM, LMPath = '../../../Bias', threshold = 0.85)
    lm_preds = {}
    
    with open(testFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter = '\t')
        
        for tweet in tweetreader:
            text = tweet[1].lower().strip()
            cur_id = int(tweet[0])
            lm_pred, lm_prob = lm.predict(text)
            
            lm_preds[cur_id] = lm_pred
        
    return lm_preds
    



def main(trainFile, testFile, train_test, out_file = None, LM = None):

    #tokenize = lambda x: nltk.word_tokenize(convertChars(x).lower())
    tokenize = lambda x: nltk.word_tokenize(x.lower())
    #tokenize = lambda x: nltk.word_tokenize(' '.join(wordsegment.segment(x)).lower())

    TEXT = Field(sequential = True, tokenize = tokenize, lower = True, include_lengths=True)
    LABEL = Field(sequential = False, use_vocab = False, dtype = torch.float)
    ID = Field(sequential = False, use_vocab = False)

    off_datafields = [('id', None), ('text', TEXT), ('label', LABEL), ('is_target', None), ('target', None)]

    trn = TabularDataset.splits(path='.', train=trainFile, format='tsv', fields=off_datafields)[0]

    tst_datafields = [('id', ID), ('text', TEXT), ('label', None)]

    tst = TabularDataset(path=testFile, format='tsv', fields = tst_datafields)
    
    TEXT.build_vocab(trn, vectors='glove.6B.200d')

    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator = BucketIterator(trn, batch_size = BATCH_SIZE, device = device, sort_key = lambda x: len(x.text), sort_within_batch = True)

    test_iterator = Iterator(tst, batch_size=BATCH_SIZE, sort_key = lambda x: len(x.text), sort_within_batch=True)

    
    if(train_test == 'train' or train_test == 'both'):

        # train CNN first
        # MIDAS CNN use 256 filters, with 2, 3, 4 as filter_sizes, dropout = 0.3

        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 200
        N_FILTERS = 256
        FILTER_SIZES = [2,3,4]
        OUTPUT_DIM = 1
        DROPOUT = 0.3
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
        
        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)


        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 3

        best_valid_loss = float('inf')

        print('training CNN')
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = train_cnn(model, train_iterator, optimizer, criterion)
            #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            #if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        torch.save(model, 'MIDAS_CNN.pt')
        
        
        # train BLSTM next
        # 
        
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 200
        HIDDEN = 64
        OUTPUT_DIM = 1
        DROPOUT = 0.2
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = BLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN, OUTPUT_DIM, DROPOUT, PAD_IDX)
        
        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)


        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 3

        best_valid_loss = float('inf')

        print('training BLSTM')
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = train_blstm(model, train_iterator, optimizer, criterion)
            #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            #if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        torch.save(model, 'MIDAS_BLSTM.pt')
        
        
        # train BLSTM-GRU last
        # 

        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 200
        HIDDEN = 64
        OUTPUT_DIM = 1
        DROPOUT = 0.3
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = BLSTM_GRU(INPUT_DIM, EMBEDDING_DIM, HIDDEN, OUTPUT_DIM, DROPOUT, PAD_IDX)
        
        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)


        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 3

        best_valid_loss = float('inf')

        print('training BLSTM-GRU')
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = train_blstm(model, train_iterator, optimizer, criterion)
            #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            #if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        torch.save(model, 'MIDAS_BLSTM-GRU.pt')
        


    elif(train_test == 'test' or train_test == 'both'):
        
        #load three models, get predictions, let vote on final prediction
        if torch.cuda.is_available():                                                                                                                                      
            print ("Using cuda")                                                                                                                                           
            model = torch.load('MIDAS_CNN.pt')                                                                                                                         
        else:                                                                                                                                                              
            print("Using cpu")                                                                                                                                             
            model = torch.load('MIDAS_CNN.pt', map_location = 'cpu')     
        

        cnn_predictions, cnn_probs = test_cnn(model, test_iterator)
        
        cnn_votes = {}
        
        for id, pred in cnn_probs:
            cnn_votes[str(id)] = pred

        
        #BLSTM
        if torch.cuda.is_available():                                                                                                                                      
            print ("Using cuda")                                                                                                                                           
            model = torch.load('MIDAS_BLSTM.pt')                                                                                                                         
        else:                                                                                                                                                              
            print("Using cpu")                                                                                                                                             
            model = torch.load('MIDAS_BLSTM.pt', map_location = 'cpu')     
        
  
        blstm_predictions, blstm_probs = test_blstm(model, test_iterator)
        
        blstm_votes = {}
        
        for id, pred in blstm_probs:
            blstm_votes[str(id)] = pred

        
        #BLSTM_BGRU
        if torch.cuda.is_available():                                                                                                                                      
            print ("Using cuda")                                                                                                                                           
            model = torch.load('MIDAS_BLSTM-GRU.pt')                                                                                                                       
        else:                                                                                                                                                              
            print("Using cpu")                                                                                                                                             
            model = torch.load('MIDAS_BLSTM-GRU.pt', map_location = 'cpu')     
  
        blstm_gru_predictions, blstm_gru_probs = test_blstm(model, test_iterator)
        
        blstm_gru_votes = {}
        
        for id, pred in blstm_gru_probs:
            blstm_gru_votes[str(id)] = pred


        # if using LM, get probs
        if(LM):
            lm_preds = getLMPreds(testFile, LM)
        
        
        if(out_file):
            output = open(out_file, 'w')
        else:
            output = open('MIDAS_predictionsOut', 'w')        
        


        # have each system vote for prediction (0 or 1) if at least 2 vote for 1 score will be 2 or higher, if at least 2 vote for 0 score will be less than 2
        for id in cnn_votes:
            #votes = cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id]
            #if(votes >= 2):
            #    pred = 1
            #else:
            #    pred = 0
            pred = int(round((cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id])/3))

            # check LM and flip if needed
            if(LM):
                if(pred == 1 and lm_preds[int(id)] == 1):
                    pred = 0

            output.write(str(id) + ',' + str(pred) + '\n')

        output.close()

        


if(__name__ == "__main__"):
    sys.path.append('../../../Bias/')
    from BERT_NSP import CommunityLM, BertForNextSentencePrediction

    if(len(sys.argv) == 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif(len(sys.argv) == 5):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
