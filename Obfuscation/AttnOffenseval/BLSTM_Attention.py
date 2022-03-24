import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
from torchtext.data import Field, ReversibleField, Example
from torchtext.data import TabularDataset, Dataset
from torchtext.data import Iterator, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from itertools import chain
from torchtext.vocab import GloVe



class Masker():
    
    def __init__(self, method = 'attention', target = 'OFF', freq_thresh = 5, train_data = 'offenseval-training-v1.tsv', encoder = 'Encoder.pt', classifier = 'Classifier.pt'):

        self.method = method
        
        if(method == 'attention'):
            self.tokenize = lambda x: nltk.word_tokenize(x.lower())

            self.TEXT = Field(batch_first = True, tokenize = self.tokenize, lower = True, include_lengths=True)
            self.LABEL = Field(sequential = False)
            self.ID = Field(sequential = False, use_vocab = False)

            off_datafields = [('id', None), ('text', self.TEXT), ('label', self.LABEL), ('is_target', None), ('target', None)]

            trn = TabularDataset.splits(path='.', train=train_data, format='tsv', fields=off_datafields)[0]
    
            self.EMBEDDING_DIM = 200
            self.TEXT.build_vocab(trn, vectors=GloVe(name='6B', dim=self.EMBEDDING_DIM))
            self.LABEL.build_vocab(trn)
            
            self.BATCH_SIZE = 64
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if torch.cuda.is_available():
                print ("Using cuda")
                self.classifier = torch.load(classifier)
                self.encoder = torch.load(encoder)
                self.classifier = self.classifier.cuda()
                self.encoder = self.encoder.cuda()
            else:
                print("Using cpu")
                self.classifier = torch.load(classifier, map_location = 'cpu')
                self.encoder = torch.load(encoder, map_location = 'cpu')
                
            
        elif(method == 'freq-ratio'):
            # calculates frequencies of tokens for a target attribute
            frequencies = {}

            # smoothing parameter 
            lamb = 1
 
            self.tokenize = lambda x: nltk.word_tokenize(x.lower())
            self.freq_thresh = freq_thresh
            self.target = target

            train_csv = csv.reader(open(train_data, 'r'), delimiter = '\t')
            
            for example in train_csv:
                text = self.tokenize(example[1])
                for token in text:
                    if token not in frequencies:
                        # stores frequencies of token appearing in target attribute and not in target attribute
                        frequencies[token] = [lamb,lamb]

                    if example[2] == self.target:
                        frequencies[token][0] += 1
                    else:
                        frequencies[token][1] += 1
            
            self.salience = {}
            for token in frequencies:
                self.salience[token] = frequencies[token][0]/frequencies[token][1]



    # takes in an input sentence and masks words based on the attention,
    # those words higher than the average attention are masked.
    def attention_mask(self, input_text):
        input_fields = [('id', self.ID), ('text', self.TEXT), ('label', self.LABEL)]
        input_example = Example()
        input_example = input_example.fromlist([1, input_text, 1], input_fields)
        input_dataset = Dataset([input_example], input_fields)
        input_iter = BucketIterator(input_dataset, batch_size = 1, device = self.device, repeat = False)

        _, attn = test_model(self.encoder, self.classifier, 1, input_iter)

        attn = attn[0]
        masked_text = self.tokenize(input_text)
        avg = sum(attn)/len(attn)
        for i in range(len(attn)):
            if(attn[i] > avg):
                masked_text[i] = '[MASK]'

        #print(attn)
        return masked_text
        
        
    def freq_mask(self, input_text):
        masked_text = self.tokenize(input_text)
        for i in range(len(masked_text)):
            token = masked_text[i]

            # if token has a salience score, check again threshold 
            if token in self.salience:
                if self.salience[token] >= self.freq_thresh:
                    masked_text[i] = '[MASK]'
        
        return masked_text


    def mask(self, input_text):
        if(self.method == 'attention'):
            return self.attention_mask(input_text)
        elif(self.method == 'freq-ratio'):
            return self.freq_mask(input_text)
            
    # takes in input text and returns corresponding attentions and prediction
    def predict(self, input_text):
        input_fields = [('id', self.ID), ('text', self.TEXT), ('label', self.LABEL)]
        input_example = Example()
        input_example = input_example.fromlist([1, input_text, 1], input_fields)
        input_dataset = Dataset([input_example], input_fields)
        input_iter = BucketIterator(input_dataset, batch_size = 1, device = self.device, repeat = False)

        pred, attn, prob = test_model(self.encoder, self.classifier, 1, input_iter)
        
        pred = pred[0][1]
        prob = prob[0][1]
        attn = attn[0].tolist()
        attn = [i[0] for i in attn]
        return pred, prob, attn

        



# Encoder
# Example from https://github.com/nn116003/self-attention-classification/blob/master/imdb_attn.py
class EncoderRNN(nn.Module):


    def __init__(self, emb_dim, h_dim, v_size, gpu=True, v_vec=None, batch_first=True, pad_idx=None):
        super(EncoderRNN, self).__init__()
        self.gpu = torch.cuda.is_available()
        self.h_dim = h_dim
        self.embed = nn.Embedding(v_size, emb_dim, padding_idx = pad_idx)
        if v_vec is not None:
            self.embed.weight.data.copy_(v_vec)
        self.lstm = nn.LSTM(emb_dim, h_dim, batch_first=batch_first,
                            bidirectional=True)
        

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu and torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)
 

    def forward(self, sentence, lengths = None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.embed(sentence)
        packed_emb = emb

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)

        out, hidden = self.lstm(packed_emb, self.hidden)
        
        if lengths is not None:
            out = nn.utils.rnn.pad_packed_sequence(output)[0]
        
        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
        return out

 

class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24,1))

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.contiguous().view(-1, self.h_dim)) # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2) # (b*s, 1) -> (b, s, 1)


class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(h_dim)
        self.main = nn.Linear(h_dim, c_num)
        
        
    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs) #(b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1) # (b, s, h) -> (b, h)
        return F.log_softmax(self.main(feats)), attns



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc



def train_model(encoder, classifier, epoch, train_iter, optimizer, batch_size, log_interval=10):
    encoder.train()
    classifier.train()
    correct = 0
    
    for idx, batch in enumerate(train_iter):
        (x, x_l), y = batch.text, batch.label - 1
        #print(y)
        optimizer.zero_grad()
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

        #print(output)

        pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        correct += int(pred.eq(y.data.view_as(pred)).cpu().sum())
        #print(correct)
        if idx % log_interval == 0:
            print('train epoch: {} [{}/{}], acc:{}, loss:{}'.format(
                epoch, idx*len(x), len(train_iter)*batch_size,
                correct/float(log_interval * len(x)),
                loss.data))
            correct = 0



def test_model(encoder, classifier, epoch, test_iter):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    final_preds = []
    pred_probs = []
    attentions = []
    softmax = torch.nn.Softmax()

    for idx, batch in enumerate(test_iter):
        (x, x_l), y = batch.text, batch.label - 1
        
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        pred = output.data.max(1, keepdim=True)[1]
        
        for i in range(len(batch.id)):
            soft_probs = softmax(output[i])
            final_preds.append((batch.id[i].item(), pred[i].item()))
            pred_probs.append((batch.id[i].item(), soft_probs[1].item()))
            #print(attn[i])
            attentions.append(attn[i])

    return final_preds, attentions, pred_probs

        #correct += int(pred.eq(y.data.view_as(pred)).cpu().sum())
        #print(correct)
        #total += int(len(batch))
        #print(total)

    #print('test epoch:{}, acc:{}'.format(epoch, correct/float(total)))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def main(trainFile, testFile, train_test, out_file = None):

    tokenize = lambda x: nltk.word_tokenize(x.lower())

    TEXT = Field(batch_first = True, tokenize = tokenize, lower = True, include_lengths=True)
    LABEL = Field(sequential = False)
    ID = Field(sequential = False, use_vocab = False)


    off_datafields = [('id', None), ('text', TEXT), ('label', LABEL), ('is_target', None), ('target', None)]

    trn = TabularDataset.splits(path='.', train=trainFile, format='tsv', fields=off_datafields)[0]
        
    #for item in trn:
    #    print(trn)


    tst_datafields = [('id', ID), ('text', TEXT), ('label', LABEL)]


    tst = TabularDataset(path=testFile, format='tsv', fields = tst_datafields)

    for item in tst:
        print(item)

    EMBEDDING_DIM = 200

    TEXT.build_vocab(trn, vectors=GloVe(name='6B', dim=EMBEDDING_DIM))
    LABEL.build_vocab(trn)
    
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator = BucketIterator(trn, batch_size = BATCH_SIZE, device = device, repeat = False)#sort_key = lambda x: len(x.text), sort_within_batch = True)

    test_iterator = BucketIterator(tst, batch_size=BATCH_SIZE, device = device, repeat = False)#sort_key = lambda x: len(x.text), sort_within_batch=True)

    
    if(train_test == 'train' or train_test == 'both'):
        
        # train BLSTM 
        # 
        
        INPUT_DIM = len(TEXT.vocab)
        HIDDEN = 64
        OUTPUT_DIM = 2
        DROPOUT = 0.2
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        encoder = EncoderRNN(EMBEDDING_DIM, HIDDEN, INPUT_DIM, gpu=torch.cuda.is_available(), v_vec = TEXT.vocab.vectors, pad_idx = PAD_IDX)
        
        classifier = AttnClassifier(HIDDEN, OUTPUT_DIM)

        
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        encoder.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        encoder.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


        # init model
        def weights_init(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
                nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            
        for m in encoder.modules():
            print(m.__class__.__name__)
            weights_init(m)

        for m in classifier.modules():
            print(m.__class__.__name__)
            weights_init(m)


        
        optimizer = optim.Adam(chain(encoder.parameters(),classifier.parameters()), lr=0.001)

        encoder =  encoder.to(device)
        classifier = classifier.to(device)



        N_EPOCHS = 5

        best_valid_loss = float('inf')

        print('training BLSTM')
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            #train_loss, train_acc =
            train_model(encoder, classifier, epoch + 1, train_iterator, optimizer, BATCH_SIZE)
            #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            #test_model(encoder, classifier, epoch + 1, test_iterator)
        
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            #if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            #print(f'\tTrain Loss: {train_loss:.3f}')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        torch.save(encoder, 'Encoder.pt')
        torch.save(classifier, 'Classifier.pt')

    elif(train_test == 'test' or train_test == 'both'):
        
        #BLSTM
        encoder = torch.load('Encoder.pt')
        classifier = torch.load('Classifier.pt')

        preds, _ = test_model(encoder, classifier, 1, test_iterator)
        
        pred_dict = {}
        for id, pred in preds:
            pred_dict[str(id)] = pred


        if(out_file):
            output = open(out_file, 'w')
        else:
            output = open('BLSTM_Attention_predictionsOut', 'w')
            
        for id in pred_dict:
            pred = pred_dict[id]
            output.write(str(id) + ',' + str(pred) + '\n')


        output.close()


if(__name__ == "__main__"):
    if(len(sys.argv) == 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

