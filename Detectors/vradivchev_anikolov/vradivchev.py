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
import json
import string

# Example from https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
class BertForSequenceClassification(nn.Module):
  
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)#, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


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


# load a clean dataset
def load_dataset(trFile=None, teFile=None):
    labelsAsNums = {}
    numsAsLabels = {}
    labelNum = 0
    numTweets = 0
    testTweets = []

    x_train = []
    y_train = []
    x_test = []

    max_sequence_length = -1
    wordsegment.load()
    

    if(trFile):
        #load in train tweets and corresponding labels
        with open(trFile, 'r') as csvfile:
            tweetreader = csv.reader(csvfile, delimiter='\t')
            for tweet in tweetreader:
                text = tweet[1].strip()
                
                #replace any non standard characters with standard
                #text = convertChars(text)

                # 'The symbols ’@’ and ’#’ were excluded from the list due to their specific semantics in tweets.' - vradivchev
                text = text.replace('#', '')
                text = text.replace('@', '')

                # 'All occurrences of tokens beginning with a hashtag were split into the separate words
                # comprising the token, provided that each separate word is uppercased ' - vradivchev
                text = ' '.join(wordsegment.segment(text))

                # 'Afterwards the tweets were subjected to tokenization and lowercasing' - vradivchev
                text = " ".join(nltk.word_tokenize(text))
                text = text.lower()

                # 'Afterwards we proceeded with removing a variety of different stop words' - vradivchev
                stop_words = set(stopwords.words('english')) 
                text = ' '.join([w for w in text.split() if not w in stop_words]) 

                x_train.append(text)
                if(len(text.split()) > max_sequence_length):
                    max_sequence_length = len(text.split())

                if tweet[2] not in labelsAsNums:
                    labelsAsNums[tweet[2]] = labelNum
                    numsAsLabels[str(labelNum)] = tweet[2]
                    labelNum += 1
                y_train.append(labelsAsNums[tweet[2]])


    #load in test tweets and corresponding labels
    if(teFile):
        with open(teFile, 'r') as csvfile:
            tweetreader = csv.reader(csvfile, delimiter='\t')
            for tweet in tweetreader:
                text = tweet[1].strip()

                #text = convertChars(text)
                # 'The symbols ’@’ and ’#’ were excluded from the list due to their specific semantics in tweets.' - vradivchev
                text = text.replace('#', '')
                text = text.replace('@', '')

                # 'All occurrences of tokens beginning with a hashtag were split into the separate words
                # comprising the token, provided that each separate word is uppercased ' - vradivchev
                text = ' '.join(wordsegment.segment(text))

                # 'Afterwards the tweets were subjected to tokenization and lowercasing' - vradivchev
                text = " ".join(nltk.word_tokenize(text))
                text = text.lower()

                # 'Afterwards we proceeded with removing a variety of different stop words' - vradivchev
                stop_words = set(stopwords.words('english')) 
                text = ' '.join([w for w in text.split() if not w in stop_words]) 

                testTweets.append(tweet)
                x_test.append(text)


    return x_train, y_train, x_test, labelNum, testTweets, labelsAsNums, numsAsLabels, max_sequence_length


# a vradivchev class object which allows single query test, assumes model was pretrained and the model is stored.
class vradivchev:
    
    def __init__(self, train_data = 'offenseval-training-v1.tsv', trained_model = 'vradivchev.pt', params_file = 'vradivchev_params.json'):
        #x_train, y_train, x_test, labelNum, testTweets, labelsAsNums, numsAsLabels, max_seq_length = load_dataset(train_data)

        # load in params
        params_in = open(params_file)
        params_lines = params_in.readlines()
        params = json.loads(params_lines[0])

        self.labelNum = params['labelNum']
        self.labelsAsNums = params['labelsAsNums']
        self.numsAsLabels = params['numsAsLabels']
        self.max_seq_length = params['max_seq_length']
        
        # Load pre-trained tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # load pre-trained model
        self.model = torch.load(trained_model)

        wordsegment.load()
    
    
    # preprocess test_query
    def preprocessQuery(self, test_query):
        text = test_query.strip()
        
        # 'The symbols ’@’ and ’#’ were excluded from the list due to their specific semantics in tweets.' - vradivchev
        text = text.replace('#', '')
        text = text.replace('@', '')
        
        # 'All occurrences of tokens beginning with a hashtag were split into the separate words
        # comprising the token, provided that each separate word is uppercased ' - vradivchev
        text = ' '.join(wordsegment.segment(text))
        
        # 'Afterwards the tweets were subjected to tokenization and lowercasing' - vradivchev
        text = " ".join(nltk.word_tokenize(text))
        text = text.lower()
        
        # 'Afterwards we proceeded with removing a variety of different stop words' - vradivchev
        stop_words = set(stopwords.words('english')) 
        text = ' '.join([w for w in text.split() if not w in stop_words]) 
        
        return text



    # convert pre-processed test query into tokenized review            
    def tokenizeQuery(self, text):
        tokenized_review = self.tokenizer.tokenize(text)
            
        if len(tokenized_review) > self.max_seq_length:
            tokenized_review = tokenized_review[:self.max_seq_length]

        ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (self.max_seq_length - len(ids_review))
            
        ids_review += padding
        
        assert len(ids_review) == self.max_seq_length

        return ids_review


    def predict(self, test_query):
        
        text = self.preprocessQuery(test_query)
        ids_review = self.tokenizeQuery(text)

        test = []
        test.append(ids_review)

        
        self.model.eval()
        # make prediction
        predicted = []
        probs = []

        tests = torch.tensor(test)
        outputs = self.model(tests)
        outputs = outputs.detach().cpu().numpy().tolist()
        
        for cur_output in outputs:
            predicted.append(cur_output.index(max(cur_output)))
            # append positive class probs
            probs.append(cur_output[self.labelsAsNums['OFF']])

        # should only be length 1 since only one query was sent in
        predicted_label = self.numsAsLabels[str(predicted[0])]
        label_prob = probs[0]

        return predicted_label, label_prob


    # allows multiple queries to be tested at once.
    def predictMultiple(self, test_queries):
        
        test = []
        for test_query in test_queries:
            text = self.preprocessQuery(test_query)
            ids_review = self.tokenizeQuery(text)
            test.append(ids_review)

        
        self.model.eval()
        # make prediction
        predicted = []
        probs = []

        tests = torch.tensor(test)
        outputs = self.model(tests)
        outputs = outputs.detach().cpu().numpy().tolist()
        
        for cur_output in outputs:
            predicted.append(self.numsAsLabels[str(cur_output.index(max(cur_output)))])
            # append positive class probs
            probs.append(cur_output[self.labelsAsNums['OFF']])

        return predicted, probs



class text_dataset(Dataset):

    def __init__(self,x_y_list, max_seq_length, transform=None):
        
        self.x_y_list = x_y_list
        self.max_seq_length = max_seq_length
        self.transform = transform
        
        
    def __getitem__(self,index):
        
        max_seq_length = self.max_seq_length
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[1][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])


def train_model(model, criterion, optimizer, scheduler, dataloaders_dict, device, num_epochs=3):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        model.train()  # Set model to training mode
        
        running_loss = 0.0
            
        sentiment_corrects = 0
            
        phase = 'train'
        # Iterate over data.
        for inputs, labels in dataloaders_dict[phase]:
            #inputs = inputs
            #print(len(inputs),type(inputs),inputs)
            #inputs = torch.from_numpy(np.array(inputs)).to(device) 
            inputs = inputs.to(device) 

            sentiment = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            #print(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    print("Finished training")
    return model



def main(trainFile, testFile, train_test, out_file = None):
    #trainFile = 'offenseval-training-v1.tsv'
    #testFile = 'testset-taska_part1.tsv'
    #trainFile = 'small_offenseval-training-v1.tsv'
    #testFile = 'small_testset-taska.tsv'
    
    if(train_test == 'train' or train_test == 'both'):
        x_train, y_train, x_test, labelNum, testTweets, labelsAsNums, numsAsLabels, max_seq_length = load_dataset(trainFile, testFile)
    else:
        # load in params
        params_in = open('vradivchev_params.json')
        params_lines = params_in.readlines()
        params = json.loads(params_lines[0])

        labelNum = params['labelNum']
        labelsAsNums = params['labelsAsNums']
        numsAsLabels = params['numsAsLabels']
        max_seq_length = params['max_seq_length']
        _, _, x_test, _, testTweets, _, _, _ = load_dataset(None, testFile)


    batch_size = 16
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)

    if(train_test == 'train' or train_test == 'both'):
        train_lists = [x_train, y_train]

        training_dataset = text_dataset(x_y_list = train_lists, max_seq_length = max_seq_length)
        
        dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)}
        dataset_sizes = {'train':len(train_lists[0])}

        num_labels = labelNum + 1
        config = BertConfig()#vocab_size_or_config_json_file=32000, hidden_size=768,
        #num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        model = BertForSequenceClassification(config, num_labels)
        lrlast = .001
        lrmain = .00001
        optim1 = optim.Adam(
            [
                {"params":model.bert.parameters(),"lr": lrmain},
                {"params":model.classifier.parameters(), "lr": lrlast},
                
            ])

        #optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
        # Observe that all parameters are being optimized
        optimizer_ft = optim1
        criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders_dict, device, num_epochs=3)

        torch.save(model_ft1, 'vradivchev.pt')

        # create a json to store dictionaries and information for loading use.
        out_info = {'labelNum':labelNum, 'labelsAsNums':labelsAsNums, 'numsAsLabels':numsAsLabels, 'max_seq_length':max_seq_length}
        
        outtmp = json.dumps(out_info)

        outfile = open('vradivchev_params.json', 'w')
        outfile.write(outtmp)
        outfile.close()

    if(train_test == 'test' or train_test == 'both'):

        model_ft1 = torch.load('vradivchev.pt')
        model_ft1.eval()

        # load test set for predictions
        tests = []
        for cur_x in x_test:
            
            tokenized_review = tokenizer.tokenize(cur_x)
            
            if len(tokenized_review) > max_seq_length:
                tokenized_review = tokenized_review[:max_seq_length]

            ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

            padding = [0] * (max_seq_length - len(ids_review))
            
            ids_review += padding
        
            assert len(ids_review) == max_seq_length

            #print(ids_review)
            #ids_review = torch.tensor(ids_review)

            tests.append(ids_review)


        if(out_file == None):
            prediction_file = open('vradivchev_predictionsOut' , 'w')
        else: 
            prediction_file = open(out_file , 'w')
 
        predicted = []
        #for cur_test in tests:
        #    print(cur_test)
        #    output = model_ft1(cur_test)
        #    output = output.detach().cpu().numpy().tolist()
        #    output = output[0]
        #    
        #    predicted.append(output.index(max(output)))

        tests = torch.tensor(tests)
        outputs = model_ft1(tests)
        outputs = outputs.detach().cpu().numpy().tolist()
        vals = []
        #print(outputs)
        for cur_output in outputs:
            predicted.append(cur_output.index(max(cur_output)))
            vals.append(max(cur_output))

        confidence_file = open('vradivchev_confidenceVals.csv', 'w')
        # write predictions to file
        for j in range(len(predicted)):
            prediction_file.write(str(testTweets[j][0]) + ',' + numsAsLabels[str(predicted[j])] + '\n')
            if(numsAsLabels[str(predicted[j])] == 'OFF'):
                confidence_file.write(str(testTweets[j][0]) + ',' + str(vals[j]) + '\n')

        prediction_file.close()
        confidence_file.close()




if(__name__ == "__main__"):
    if(len(sys.argv) == 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

