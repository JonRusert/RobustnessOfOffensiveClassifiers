# Obfuscator 

import csv
import sys
import random
sys.path.append('../Detectors/NULI/')
sys.path.append('../Detectors/vradivchev_anikolov/')
sys.path.append('../Detectors/MIDAS/')
sys.path.append('../Detectors/GooglePerspective/')
sys.path.append('../Detectors/LexiconDetect/')
sys.path.append('AttnOffenseval/')
from BLSTM_Attention import Masker, EncoderRNN, Attn, AttnClassifier
from NULI import NULI, BertForSequenceClassification
from vradivchev import vradivchev, BertForSequenceClassification
from MIDAS import MIDAS, CNN, BLSTM, BLSTM_GRU
from Perspective import Perspective
from LexiconDetect import LexiconDetect
import sent2vec
import numpy as np
import json
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import emoji
import wordsegment
wordsegment.load()
from nltk.corpus import wordnet, words
import nltk
from nltk.tokenize import TweetTokenizer


class Obfuscator:
    
    def __init__(self, obf_method = 'GS_GR', detection_alg = 'NULI', embedding_file = 'glove/glove.twitter.27B.100d.txt', sent_emb_file = 'sent2vecModels/twitter_bigrams.bin', filter_min = 3, filtered_embs = None, alg_version = 0, persp_key = 1):
        self.obf_method = obf_method
        
        if(detection_alg == 'NULI'):
            self.detector = NULI(train_data= '../Detectors/NULI/offenseval-training-v1.tsv', trained_model = '../Detectors/NULI/NULI.pt', params_file = '../Detectors/NULI/NULI_params.json')
        elif(detection_alg == 'vradivchev'):
            self.detector = vradivchev(train_data= '../Detectors/vradivchev_anikolov/offenseval-training-v1.tsv', trained_model = '../Detectors/vradivchev_anikolov/vradivchev.pt', params_file = '../Detectors/vradivchev_anikolov/vradivchev_params.json')
        elif(detection_alg == 'MIDAS'):
            self.detector =  MIDAS(train_data= '../Detectors/MIDAS/offenseval-training-v1.tsv', trained_cnn_model = '../Detectors/MIDAS/MIDAS_CNN.pt', trained_blstm_model = '../Detectors/MIDAS/MIDAS_BLSTM.pt', trained_blstmGru_model = '../Detectors/MIDAS/MIDAS_BLSTM-GRU.pt')
        elif(detection_alg == 'Perspective'):
            self.detector = Perspective(threshold = 0.5, select = persp_key)
        elif(detection_alg == 'LexiconDetect'):
            self.detector = LexiconDetect(offensive_word_file = '../Detectors/LexiconDetect/abusive_words.txt')
        else:
            self.detector = None
            print('Detection algorithm not available, or not using detector.')
        
        # load in embeddings, differs for methods as GR uses word embeddings and EC uses sentence embeddings
        if(obf_method == 'GS_GR'):
            self.embeddingDict = {}
            embeddings = open(embedding_file, 'r')

            for embedding in embeddings:
                embedding = embedding.strip().split()
                self.embeddingDict[embedding[0]] = [float(x) for x in embedding[1:]]
                
            embeddings.close()

        elif(obf_method == 'GS_EC' or obf_method == 'GS_EC_MAX' or obf_method == 'GS_EC_MAX_Single' or obf_method == 'GS_EC_MAX_Extended' or obf_method == 'GSS_BF_MAX' or obf_method == 'GSS_EC_MAX'):
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(sent_emb_file)
            
            # convert glove embedding to word2vec embedding
            #glove_file = datapath(os.path.abspath())
            #tmp_file = get_tmpfile(os.path.abspath('glove/word2vec.twitter.27B.100d.txt'))
            #_ = glove2word2vec(glove_file, tmp_file)
            self.w2v_model = KeyedVectors.load_word2vec_format(embedding_file)
        
        elif(obf_method == 'AT_EC_MAX'):
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(sent_emb_file)
            
            # convert glove embedding to word2vec embedding
            #glove_file = datapath(os.path.abspath())
            #tmp_file = get_tmpfile(os.path.abspath('glove/word2vec.twitter.27B.100d.txt'))
            #_ = glove2word2vec(glove_file, tmp_file)
            self.w2v_model = KeyedVectors.load_word2vec_format(embedding_file)
            
            self.masker = Masker(method = 'attention', target = 'OFF', freq_thresh = 5, train_data = 'AttnOffenseval/offenseval-training-v1.tsv', encoder = 'AttnOffenseval/Encoder.pt', classifier = 'AttnOffenseval/Classifier.pt')
            self.tokenize = lambda x: nltk.word_tokenize(x.lower())
            

        
        # refers to which algorithm should be used with the filter embbeding file
        # 0 = remove all with less frequency than filter_min (note if no file is passed for filtered_embs, no words are removed)
        # 1 = 0 + sort candidate words by highest frequency
        # 2 = 0 + sort candidate words by lowest frequency
        # 3 = 0 + rerank candidate list by sim_pos + freq_pos where sim_pos = 1 if most_similar and freq_pos = 1 if most frequent
        self.alg_version = int(alg_version)
        '''
        # if filter embedding files, read in and store
        filter_min = int(filter_min)
        self.filtered_embs = {}
        if(filtered_embs):
            filtercsv = csv.reader(open(filtered_embs), delimiter = ',')
            for cur_line in filtercsv:
                word = cur_line[0].strip()
                self.filtered_embs[word] = []
                
                if(self.alg_version == 0):
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            self.filtered_embs[word].append(sim.strip())

                elif(self.alg_version == 1):
                    tmp_sims = {}
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            tmp_sims[sim.strip()] = int(count)
                        
                    self.filtered_embs[word].extend(sorted(tmp_sims, key=tmp_sims.get, reverse = True))

                elif(self.alg_version == 2):
                    tmp_sims = {}
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            tmp_sims[sim.strip()] = int(count)
                        
                    self.filtered_embs[word].extend(sorted(tmp_sims, key=tmp_sims.get))
                
                else: 
                    tmp_sims = {}
                    tmp_pos = {}
                    cur_pos = 1
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            tmp_sims[sim.strip()] = int(count)
                            tmp_pos[sim.strip()] = cur_pos
                            cur_pos += 1
                    
                    tmp_freqs = {}
                    cur_freq = 1
                    for cur in sorted(tmp_sims, key = tmp_sims.get, reverse = True):
                        tmp_freqs[cur] = cur_freq
                        cur_freq += 1
                        
                    # join sim and freq rankings together
                    final_ranks = {}
                    for cur in tmp_pos:
                        final_ranks[cur] = tmp_pos[cur] + tmp_freqs[cur]
                        
                    self.filtered_embs[word].extend(sorted(final_ranks, key=final_ranks.get))
                    
        #print(self.filtered_embs['dishonest'])
        '''
        # refers to which algorithm should be used with the filter embbeding file
        # 0 = remove all with less frequency than filter_min (note if no file is passed for filtered_embs, no words are removed)
        # 1 = 0 + sort candidate words by highest frequency
        # 2 = 0 + sort candidate words by lowest frequency
        # 3 = 0 + rerank candidate list by sim_pos + freq_pos where sim_pos = 1 if most_similar and freq_pos = 1 if most frequent
        self.alg_version = int(alg_version)
        
        # if filter embedding files, read in and store
        self.filter_min = int(filter_min)
        self.filtered_embs = {}
        if(filtered_embs):
            filtercsv = csv.reader(open(filtered_embs), delimiter = ',')
            for cur_line in filtercsv:
                cur_word = cur_line[0]
                cur_freq = int(cur_line[1])
                self.filtered_embs[cur_word] = cur_freq 
        #'''

    def selectMethod(self, obf_method):
        self.obf_method = obf_method


    def preProcessText(self, text):
        
        text = text.lower().strip()

        ## change emojis to readable text
        ## segment text
        #text = ' '.join(wordsegment.segment(emoji.demojize(text)))
        #text = ' '.join(wordsegment.segment(text))

        tknzr = TweetTokenizer()
        text = ' '.join(tknzr.tokenize(text))

        return text


    def postProcessText(self, text, processed_text):
        print(text)
        orig_text = text.strip().split()
        print(orig_text)
        processed_text = processed_text.split()
        print(processed_text)

        tknzr = TweetTokenizer()
        tokenized_text = ' '.join(tknzr.tokenize(text.strip()))
        tokenized_text = tokenized_text.split()
        print(tokenized_text)

        mapping = {}
        j = 0
        i = 0

        cur_seq = ''
        cur_seq_is = []


        # map to tokenized text
        while(i < len(tokenized_text)):
            if(j >= len(orig_text)):
                break

            cur_seq += tokenized_text[i] 

            if(cur_seq == orig_text[j]):
                for k in cur_seq_is:
                    mapping[k] = j

                mapping[i] = j
                i += 1
                j += 1
                cur_seq = ''
                continue 

            
            if(cur_seq in orig_text[j]):
                cur_seq_is.append(i)
                i += 1
            else:
                j += 1
        

        alt_mapping = {}
        j = 0
        i = 0

        # alternate map to tokenized text
        while(i < len(tokenized_text)):
            if(j >= len(orig_text)):
                break

            cur_toks = tknzr.tokenize(orig_text[j])
            
            if(tokenized_text[i] in cur_toks):
                alt_mapping[i] = j
                i += 1
            else:
                j += 1


        
        print('mapping:', mapping)
        print('alt_mapping:', alt_mapping)
        
        # use alt mapping to fill in missing info for mapping
        for i in range(len(tokenized_text)):
            if(i not in mapping):
                mapping[i] = alt_mapping[i]


        
                
        print('filled mapping:', mapping)
        # use mapping to map to new text
        i = 0
        textmapping = {}
        while(i < len(processed_text)):

            j = mapping[i]
            
            # if all uppercase, revert back
            if(orig_text[j].isupper()):
                processed_text[i] = processed_text[i].upper()
                
            # if start with capital letter
            if(orig_text[j][0].isupper()):
                processed_text[i] = processed_text[i][0].upper() + processed_text[i][1:]
                
            if(j not in textmapping):
                textmapping[j] = processed_text[i]
            else:
                textmapping[j] = textmapping[j] + processed_text[i]
                
            i += 1
        
        j = 0
        outtext = []
        print(textmapping)
        for j in sorted(textmapping):
            outtext.append(textmapping[j])
            
        return ' '.join(outtext)

                


    # determines word to be chosen via greedy select (checking probability changes) and greedy replaces (random replacement)
    def GS_GR(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations
        orig_pred, var_probs = self.detector.predictMultiple(variations)
            
        for cur_prob in var_probs:
                prob_diffs.append(initial_prob - cur_prob)
            
            
        replace_pos = prob_diffs.index(max(prob_diffs))
        
        # get a random word from vocab to replace word
        rand_pos = random.randint(0, len(self.embeddingDict))
        replace_word = list(self.embeddingDict.keys())[rand_pos]
        replaced = [rand_pos]

        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

        # if the query without the word is not offensive randomly choose until you find a non offensive replacement
        if(orig_pred[replace_pos] == 'NOT'):
            # keep randomly replacing while prediction is OFF
            new_pred, _ = self.detector.predict(obf_query)
            while(new_pred == 'OFF'):
                # if all embeddings attempted, break out
                if(len(replaced) == len(self.embeddingDict)):
                    break

                rand_pos = random.choice([x for x in range(0, len(self.embeddingDict)) if x not in replaced])
                replaced.append(rand_pos)
                replace_word = list(self.embeddingDict.keys())[rand_pos]

                obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])
                new_pred, _ = self.detector.predict(obf_query)


        print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos], 'replacement:')
        return obf_query

    
    # determines word to be chosen via greedy select (checking probability changes) and replaces using constraints on the embedding
    def GS_EC(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations

        orig_pred, var_probs = self.detector.predictMultiple(variations)
            
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)

        
        replace_pos = prob_diffs.index(max(prob_diffs))
            
        
        # find closest embedding in vocab to replace word, previous measurements are store to reduce runtime
        #l1_dict = {}
        #orig_emb = self.model.embed_sentence(query)
        #for can_word in self.vocab:
        #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
        #    new_emb = self.model.embed_sentence(cur_obf)

        #    l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
        #    l1_dict[can_word] = l1_dist

        # if the query without the word is not offensive choose minimum distance until you find a non offensive replacement
        #for can_word in sorted(l1_dict, key=l1_dict.get):
        #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
        #    obf_pred, _ = self.detector.predict(cur_obf)
        #    replace_word = can_word
        #    if(obf_pred == 'NOT' or orig_pred[replace_pos] == 'OFF'):
        #        break


        # sort by minimum l1 distance, if sentence is not offensive use, else move to next candidate word
        #print(l1_dict)

        candidate_words = []

        # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
        if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
            candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
        else:
            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[replace_pos])
            except: #if replacement word does not exist in the vocabulary generate random list
                w2v_vocab = list(self.w2v_model.vocab.keys())
                for _ in range(10):
                    rand_pos = random.randint(0, len(w2v_vocab))
                    candidate_words.append((w2v_vocab[rand_pos], 0))

            tmp_words = candidate_words.copy()
            # filter out any non words from candidate list
            #for candidate in candidate_words:
            #    temp_candidate = candidate[0]
            #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
            #        candidate_words.remove(candidate)
                
            # if no english words, use anyways
            #if len(candidate_words) == 0:
            #    candidate_words = tmp_words.copy()

        lowest_distance = float('inf')
        replace_word = ''
        found_replace = False
        cands_probs = {}
        cands_dists = {}

        orig_emb = self.model.embed_sentence(query)
        # choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
        for cand in candidate_words:
            if(type(cand) == tuple):
                can_word = cand[0]
            else:
                can_word = cand
            cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            obf_pred, obf_prob = self.detector.predict(cur_obf)
            cands_probs[can_word] = obf_prob

            new_emb = self.model.embed_sentence(cur_obf)

            l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
            cands_dists[can_word] = l1_dist

            if(l1_dist < lowest_distance and obf_pred != 'OFF'):
                lowest_distance = l1_dist
                replace_word = can_word
                found_replace = True

        # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest OFF score
        if(not found_replace):
            replace_word = sorted(cands_probs, key = cands_probs.get)[0]
                    
                
        

        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])
        
        print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos], 'l1 dist:', cands_dists[replace_word], 'replacement:', replace_word)
        return obf_query



    # determines MULTIPLE words to be chosen via greedy select (checking probability changes until no longer offensive) and replaces using constraints on the embedding
    def GS_EC_MAX_Single(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations

        orig_pred, var_probs = self.detector.predictMultiple(variations)
        
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)
        
        finished = False
        remaining_diffs = []
        for x in prob_diffs:
            remaining_diffs.append(x)

        while(not finished):
            finished = True
            
            # if all positions have been replaced, break out
            if(len(remaining_diffs) == 0):
                finished = True
                break

            
            replace_pos = prob_diffs.index(max(remaining_diffs))
            remove_pos = remaining_diffs.index(max(remaining_diffs))
            remaining_diffs.pop(remove_pos)

            # find closest embedding in vocab to replace word, previous measurements are store to reduce runtime
            #l1_dict = {}
            #orig_emb = self.model.embed_sentence(query)
            #for can_word in self.vocab:
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    new_emb = self.model.embed_sentence(cur_obf)

            #    l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
            #    l1_dict[can_word] = l1_dist

            # if the query without the word is not offensive choose minimum distance until you find a non offensive replacement
            #for can_word in sorted(l1_dict, key=l1_dict.get):
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    obf_pred, _ = self.detector.predict(cur_obf)
            #    replace_word = can_word
            #    if(obf_pred == 'NOT' or orig_pred[replace_pos] == 'OFF'):
            #        break


            # sort by minimum l1 distance, if sentence is not offensive use, else move to next candidate word
            #print(l1_dict)


            candidate_words = []

            # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
            #if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
            #    candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
            #else:
            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[replace_pos], topn = 20)
                tmp_words = candidate_words.copy()
                
                # filter out any low frequency words from candidate list
                for candidate in tmp_words:
                    temp_candidate = candidate[0]
                    if(temp_candidate in self.filtered_embs):
                        if(self.filtered_embs[temp_candidate] < self.filter_min):
                            candidate_words.remove(candidate)
                
                # make sure some substitutions exist
                if len(candidate_words) == 0:
                    print("NO CANDIDATE WORDS:", split_query[replace_pos])
                    candidate_words = tmp_words.copy()


            except: #if replacement word does not exist in the vocabulary generate random list
                print('CANNOT FIND WORD:', split_query[replace_pos])
                continue
                #w2v_vocab = list(self.w2v_model.vocab.keys())
                #for _ in range(10):
                #    rand_pos = random.randint(0, len(w2v_vocab))
                #    candidate_words.append((w2v_vocab[rand_pos], 0))

                #tmp_words = candidate_words.copy()
                # filter out any non words from candidate list
                #for candidate in candidate_words:
                #    temp_candidate = candidate[0]
                #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
                #        candidate_words.remove(candidate)

                # if no english words, use anyways
                #if len(candidate_words) == 0:
                #    candidate_words = tmp_words.copy()


            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}
            cands_single = {}

            orig_emb = self.model.embed_sentence(query)
            #print(candidates_words)

            ## choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
            ## choose candidate word which does not make message OFF
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand
            
                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                obf_pred, obf_prob = self.detector.predict(cur_obf)
                cands_probs[can_word] = obf_prob
                single_pred, single_prob = self.detector.predict(can_word)
                if(single_pred == 'NOT'):
                    cands_single[can_word] = single_prob


                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                if(l1_dist < lowest_distance and obf_pred != 'OFF'):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                 
                #if(obf_pred != 'OFF'):
                #    replace_word = can_word
                #    found_replace = True
                 
   

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which is the most OFF, not offensive
            if(not found_replace):
                if(len(cands_single) == 0):
                    replace_word = sorted(cands_probs, key = cands_probs.get)[0]
                    finished = False
                else:
                    replace_word = sorted(cands_single, key = cands_single.get, reverse = True)[0]
                    finished = False

            obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])#, 'l1 dist:', cands_dists[replace_word])

            split_query = obf_query.split()

        return obf_query


    # determines MULTIPLE words to be chosen via greedy select (checking probability changes until no longer offensive) and replaces using constraints on the embedding
    def GS_EC_MAX(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []
 
        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations

        orig_pred, var_probs = self.detector.predictMultiple(variations)
        
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)
        
        finished = False
        remaining_diffs = []
        for x in prob_diffs:
            remaining_diffs.append(x)

        while(not finished):
            finished = True
            
            # if all positions have been replaced, break out
            if(len(remaining_diffs) == 0):
                finished = True
                break

            
            replace_pos = prob_diffs.index(max(remaining_diffs))
            remove_pos = remaining_diffs.index(max(remaining_diffs))
            remaining_diffs.pop(remove_pos)

            # find closest embedding in vocab to replace word, previous measurements are store to reduce runtime
            #l1_dict = {}
            #orig_emb = self.model.embed_sentence(query)
            #for can_word in self.vocab:
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    new_emb = self.model.embed_sentence(cur_obf)

            #    l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
            #    l1_dict[can_word] = l1_dist

            # if the query without the word is not offensive choose minimum distance until you find a non offensive replacement
            #for can_word in sorted(l1_dict, key=l1_dict.get):
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    obf_pred, _ = self.detector.predict(cur_obf)
            #    replace_word = can_word
            #    if(obf_pred == 'NOT' or orig_pred[replace_pos] == 'OFF'):
            #        break


            # sort by minimum l1 distance, if sentence is not offensive use, else move to next candidate word
            #print(l1_dict)


            candidate_words = []

            # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
            #if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
            #    candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
            #else:
            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[replace_pos], topn = 20)                
                tmp_words = candidate_words.copy()
                
                # filter out any low frequency words from candidate list
                for candidate in tmp_words:
                    temp_candidate = candidate[0]
                    if(temp_candidate in self.filtered_embs):
                        if(self.filtered_embs[temp_candidate] < self.filter_min):
                            candidate_words.remove(candidate)
                
                # make sure some substitutions exist
                if len(candidate_words) == 0:
                    print("NO CANDIDATE WORDS:", split_query[replace_pos])
                    candidate_words = tmp_words.copy()


            except: #if replacement word does not exist in the vocabulary generate random list
                print('CANNOT FIND WORD:', split_query[replace_pos])
                continue
                #w2v_vocab = list(self.w2v_model.vocab.keys())
                #for _ in range(10):
                #    rand_pos = random.randint(0, len(w2v_vocab))
                #    candidate_words.append((w2v_vocab[rand_pos], 0))

                #tmp_words = candidate_words.copy()
                # filter out any non words from candidate list
                #for candidate in candidate_words:
                #    temp_candidate = candidate[0]
                #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
                #        candidate_words.remove(candidate)

                # if no english words, use anyways
                #if len(candidate_words) == 0:
                #    candidate_words = tmp_words.copy()


            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}
            cands_emb_dists = {}

            orig_emb = self.model.embed_sentence(query)
            #print(candidates_words)

            ## choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
            ## choose candidate word which does not make message OFF
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand
            
                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                obf_pred, obf_prob = self.detector.predict(cur_obf)
                cands_probs[can_word] = obf_prob
                #cands_emb_dists[can_word] = self.w2v_model.similarity(split_query[replace_pos], can_word)

                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                if(l1_dist < lowest_distance and obf_pred != 'OFF'):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                 
                #if(obf_pred != 'OFF'):
                #    replace_word = can_word
                #    found_replace = True
                 
   

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest OFF score
            if(not found_replace):
                #comb_scores = {}
                #for x in cands_probs:
                #    comb_scores[x] = (1 - cands_probs[x]) + cands_emb_dists[x]

                replace_word = sorted(cands_probs, key = cands_probs.get)[0] 
                #replace_word = sorted(comb_scores, key = comb_scores.get)[0] 
                finished = False


            obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])#, 'l1 dist:', cands_dists[replace_word])

            split_query = obf_query.split()

        return obf_query



    # determines MULTIPLE words to be chosen via greedy select (checking probability changes until no longer offensive) and replaces using constraints on the embedding
    def GSS_BF_MAX(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        
        needsReplacing = []
        done = False
        orig_pos = {}    
        split_query = query.split()
            
        #note original position so when multiple are removed, original is retained
        for i in range(len(split_query)):
            orig_pos[i] = i

        # choose tokens to be replaced before replacing any 
        while(not done):
            variations = []
            prob_diffs = []

            # step through each word and generate the variations of the original query by removing one word at a time
            for cur_pos in range(len(split_query)):
                modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
                variations.append(modified_query)

            # get probabilities for all variations

            orig_preds, var_probs = self.detector.predictMultiple(variations)

            for cur_prob in var_probs:
                prob_diffs.append(initial_prob - cur_prob)
            
            replace_pos = prob_diffs.index(max(prob_diffs))
            replace_pred = orig_preds[replace_pos]
            
            orig_repl = orig_pos[replace_pos]
            
            needsReplacing.append(orig_repl)
            split_query.pop(replace_pos)

            if(replace_pred == 'NOT'):
                done = True
            else: # update original positions
                for i in range(len(split_query)):            
                    if(i < replace_pos):
                        orig_pos[i] = orig_pos[i]
                    elif(i >= replace_pos):
                        orig_pos[i] = orig_pos[i+1]
            
            


        pos_candidates = {}
        split_query = query.split()
        
        # get candidate words for each position to be replaced
        for pos in needsReplacing:
        
            candidate_words = []

            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[pos], topn = 10)                
                tmp_words = candidate_words.copy()
                
                # filter out any low frequency words from candidate list
                for candidate in tmp_words:
                    temp_candidate = candidate[0]
                    if(temp_candidate in self.filtered_embs):
                        if(self.filtered_embs[temp_candidate] < self.filter_min):
                            candidate_words.remove(candidate)
                
                # make sure some substitutions exist
                if len(candidate_words) == 0:
                    print("NO CANDIDATE WORDS:", split_query[pos])
                    candidate_words = tmp_words.copy()

                    
            except: #if replacement word does not exist in the vocabulary generate random list
                print('CANNOT FIND WORD:', split_query[pos])
                needsReplacing.remove(pos)
                continue
                
            pos_candidates[pos] = [x[0] for x in candidate_words]
            
        
         
            
        #print(pos_candidates)
        replacementGrid = []
        # create variations of all combinations of candidate words and replacements topn^len(needsReplacing)
        for pos in needsReplacing:
            tmpGrid = []
            
            for cur_rep in pos_candidates[pos]:
                rep = {pos:cur_rep}

                if(len(replacementGrid) == 0): #first pos
                    tmpGrid.append(rep)                
                else:
                    for other in replacementGrid:
                        other.update(rep)
                        tmpGrid.append(other)
                
            replacementGrid = tmpGrid.copy()

        #print(replacementGrid)
        
        # generate variations for testing
        variations = []
        costs = {}
        j = 0
        for rep in replacementGrid:
            tmp_query = split_query.copy()
            cost = 0

            #print(rep)
            for key in rep:
                tmp_query[key] = rep[key]
                cost += self.w2v_model.similarity(split_query[key], rep[key]) #cosine similarity between original and replacement word

            costs[j] = cost
            j += 1
            tmp_var = ' '.join(tmp_query)
            variations.append(tmp_var)

            
        # get probs for all variations
        var_preds, var_probs = self.detector.predictMultiple(variations)

        # if combination results in NOT, choose most similar combination from those, otherwise choose lowest offensive prob
        if('NOT' in var_preds):
            choices = [i for i in range(len(var_preds)) if var_preds[i] == 'NOT']
            print(costs)
            print(choices)
            high = 0
            final = 0

            for i in choices:
                if(costs[i] > high):
                    high = costs[i]
                    final = i
            
        else:
            low = float('inf')
            final = 0
            
            for i in range(len(variations)):
                if(var_probs[i] < low):
                    low = var_probs[i]
                    final = i
                    
            
        return variations[final]
          
        
        
        '''
        while(not finished):
            finished = True
            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}
            cands_emb_dists = {}

            orig_emb = self.model.embed_sentence(query)
            #print(candidates_words)

            ## choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
            ## choose candidate word which does not make message OFF
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand
            
                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                obf_pred, obf_prob = self.detector.predict(cur_obf)
                cands_probs[can_word] = obf_prob
                #cands_emb_dists[can_word] = self.w2v_model.similarity(split_query[replace_pos], can_word)

                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                if(l1_dist < lowest_distance and obf_pred != 'OFF'):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                 
                #if(obf_pred != 'OFF'):
                #    replace_word = can_word
                #    found_replace = True
                 
   

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest OFF score
            if(not found_replace):
                #comb_scores = {}
                #for x in cands_probs:
                #    comb_scores[x] = (1 - cands_probs[x]) + cands_emb_dists[x]

                replace_word = sorted(cands_probs, key = cands_probs.get)[0] 
                #replace_word = sorted(comb_scores, key = comb_scores.get)[0] 
                finished = False


            #obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            #print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])#, 'l1 dist:', cands_dists[replace_word])

            #split_query = obf_query.split()

        return obf_query
        '''


    # determines MULTIPLE words to be chosen via greedy select (checking probability changes until no longer offensive) and replaces using constraints on the embedding
    def GSS_EC_MAX(self, query):
        orig_text = query
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        print(query, initial_prob)
        needsReplacing = []
        done = False
        orig_pos = {}    
        split_query = query.split()
            
        #note original position so when multiple are removed, original is retained
        for i in range(len(split_query)):
            orig_pos[i] = i

        count = 0
        # choose tokens to be replaced before replacing any 
        while(not done):
            variations = []
            prob_diffs = []

            # step through each word and generate the variations of the original query by removing one word at a time
            for cur_pos in range(len(split_query)):
                modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
                variations.append(modified_query)

            # get probabilities for all variations

            orig_preds, var_probs = self.detector.predictMultiple(variations)

            for cur_prob in var_probs:
                prob_diffs.append(initial_prob - cur_prob)
            
            if(count == 0):
                print(split_query, prob_diffs)
                count += 1

            replace_pos = prob_diffs.index(max(prob_diffs))
            replace_pred = orig_preds[replace_pos]
             
            
            orig_repl = orig_pos[replace_pos]
            
            print(split_query[replace_pos], var_probs[replace_pos])

            needsReplacing.append(orig_repl)
            split_query.pop(replace_pos)

            if(replace_pred == 'NOT'):
                done = True
            else: # update original positions
                for i in range(len(split_query)):            
                    if(i < replace_pos):
                        orig_pos[i] = orig_pos[i]
                    elif(i >= replace_pos):
                        orig_pos[i] = orig_pos[i+1]
            
            


        pos_candidates = {}
        split_query = query.split()
        #print(needsReplacing)
        tmpReplacing = needsReplacing.copy()
        # get candidate words for each position to be replaced
        for pos in needsReplacing:
        
            candidate_words = []

            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[pos], topn = 20)                
                tmp_words = candidate_words.copy()
                
                # filter out any low frequency words from candidate list
                for candidate in tmp_words:
                    temp_candidate = candidate[0]
                    if(temp_candidate in self.filtered_embs):
                        if(self.filtered_embs[temp_candidate] < self.filter_min):
                            candidate_words.remove(candidate)
                
                # make sure some substitutions exist
                if len(candidate_words) == 0:
                    print("NO CANDIDATE WORDS:", split_query[pos])
                    candidate_words = tmp_words.copy()

                    
            except: #if replacement word does not exist in the vocabulary generate random list
                print('CANNOT FIND WORD:', split_query[pos])
                tmpReplacing.remove(pos)
                continue
                
            pos_candidates[pos] = [x[0] for x in candidate_words]
            
        needsReplacing = tmpReplacing.copy()
         
            
        orig_emb = self.model.embed_sentence(query)
        
        choices = {}
        #print(pos_candidates)
        # find best fit word for each replacement by choosing the closest non offensive word
        for pos in needsReplacing:
            cands = pos_candidates[pos]
            
            replaced = split_query.copy()
            #remove other words needing to be replaced 
            for pos2 in needsReplacing:
                if(pos2 not in choices):
                    replaced[pos2] = ''
                else:
                    replaced[pos2] = choices[pos2]

            cur_probs = {}
            foundNon = False
            lowest_distance = float('inf')
            cur_dists = {}

            for cur in cands:
                replaced[pos] = cur
                
                r_comb = ' '.join(replaced)
                
                pred, prob = self.detector.predict(r_comb)
                
                cur_probs[cur] = prob
               
                #'''
                new_emb = self.model.embed_sentence(r_comb)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cur_dists[cur] = l1_dist

                if(l1_dist < lowest_distance and prob < 0.7): #pred != 'OFF'):  
                    lowest_distance = l1_dist
                    foundNon = True
                    choices[pos] = cur
                '''
                if(pred == 'NOT'):
                    choices[pos] = cur
                    foundNon = True
                    break
                #'''

            if(not foundNon): #choose least offensive
                replace_word = sorted(cur_probs, key = cur_probs.get)[0] 
                choices[pos] = replace_word
            
            # after replacement check probs again since the next candidates might not need to be replaced
            tmpQuery = split_query.copy()
            for pos2 in needsReplacing:
                if(pos2 in choices):
                    tmpQuery[pos2] = choices[pos2]
            tmpText = ' '.join(tmpQuery)
            tmppred, tmpprob = self.detector.predict(tmpText)
            print(tmpText, tmpprob)
            if(tmppred != 'OFF'):
                break # no need to finish replacing the rest of the candidates

            
        #print(choices)
        #print(needsReplacing)
        #print(split_query)
        final_query = split_query
        # make final replacements
        for pos in choices:
            final_query[pos] = choices[pos]
        
        final_text = ' '.join(final_query)

        final_text = self.postProcessText(orig_text, final_text)

        return final_text
        
    



# determines MULTIPLE words to be chosen via greedy select (checking probability changes until no longer offensive) and replaces using constraints on the embedding
    def GS_EC_MAX_Extended(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []
 
        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations

        orig_pred, var_probs = self.detector.predictMultiple(variations)
        
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)
        
        finished = False
        remaining_diffs = []
        for x in prob_diffs:
            remaining_diffs.append(x)

        while(not finished):
            finished = True
            
            # if all positions have been replaced, break out
            if(len(remaining_diffs) == 0):
                finished = True
                break

            
            replace_pos = prob_diffs.index(max(remaining_diffs))
            remove_pos = remaining_diffs.index(max(remaining_diffs))
            remaining_diffs.pop(remove_pos)

            # find closest embedding in vocab to replace word, previous measurements are store to reduce runtime
            #l1_dict = {}
            #orig_emb = self.model.embed_sentence(query)
            #for can_word in self.vocab:
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    new_emb = self.model.embed_sentence(cur_obf)

            #    l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
            #    l1_dict[can_word] = l1_dist

            # if the query without the word is not offensive choose minimum distance until you find a non offensive replacement
            #for can_word in sorted(l1_dict, key=l1_dict.get):
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    obf_pred, _ = self.detector.predict(cur_obf)
            #    replace_word = can_word
            #    if(obf_pred == 'NOT' or orig_pred[replace_pos] == 'OFF'):
            #        break


            # sort by minimum l1 distance, if sentence is not offensive use, else move to next candidate word
            #print(l1_dict)


            candidate_words = []

            # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
            #if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
            #    candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
            #else:
            # use word2vec to get list of closest words to the word to be replaced
            try:
                initial_candidate_words = self.w2v_model.most_similar(split_query[replace_pos], topn = 10)
                candidate_words.extend([cur[0] for cur in initial_candidate_words])

                for cur in initial_candidate_words:
                    cur_init = cur[0]
                    tmp_words = self.w2v_model.most_similar(cur_init, topn = 5)
                    for cur_cand in tmp_words:
                        if(cur_cand[0] not in candidate_words):
                            candidate_words.append(cur_cand[0])

                tmp_words = candidate_words.copy()
                
                # filter out any low frequency words from candidate list
                for candidate in tmp_words:
                    temp_candidate = candidate

                    if(temp_candidate in self.filtered_embs):
                        if(self.filtered_embs[temp_candidate] < self.filter_min):
                            candidate_words.remove(candidate)
                
                # make sure some substitutions exist
                if len(candidate_words) == 0:
                    print("NO CANDIDATE WORDS:", split_query[replace_pos])
                    candidate_words = tmp_words.copy()


            except: #if replacement word does not exist in the vocabulary generate random list
                print('CANNOT FIND WORD:', split_query[replace_pos])
                continue
                #w2v_vocab = list(self.w2v_model.vocab.keys())
                #for _ in range(10):
                #    rand_pos = random.randint(0, len(w2v_vocab))
                #    candidate_words.append((w2v_vocab[rand_pos], 0))

                #tmp_words = candidate_words.copy()
                # filter out any non words from candidate list
                #for candidate in candidate_words:
                #    temp_candidate = candidate[0]
                #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
                #        candidate_words.remove(candidate)

                # if no english words, use anyways
                #if len(candidate_words) == 0:
                #    candidate_words = tmp_words.copy()


            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}

            orig_emb = self.model.embed_sentence(query)
            #print(candidates_words)

            ## choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
            ## choose candidate word which does not make message OFF
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand
            
                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                obf_pred, obf_prob = self.detector.predict(cur_obf)
                cands_probs[can_word] = obf_prob

                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                if(l1_dist < lowest_distance and obf_pred != 'OFF'):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                 
                #if(obf_pred != 'OFF'):
                #    replace_word = can_word
                #    found_replace = True
                 
   

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest OFF score
            if(not found_replace):
                replace_word = sorted(cands_probs, key = cands_probs.get)[0] 
                finished = False


            obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])#, 'l1 dist:', cands_dists[replace_word])

            split_query = obf_query.split()

        return obf_query





# determines MULTIPLE words to be chosen via attention select (checking attentions until no longer offensive) and replaces using constraints on the embedding
    def AT_EC_MAX(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        pred, prob, atts = self.masker.predict(query)
        
        #split_query = self.tokenize(query)
        split_query = query.split()

        finished = False

        orig_atts = atts.copy()

        while(not finished):
            finished = True
            
            # if all positions have been replaced, break out
            if(len(atts) == 0):
                finished = True
                break

            
            replace_pos = orig_atts.index(max(atts))
            remove_pos = atts.index(max(atts))
            atts.pop(remove_pos)



            candidate_words = []

            # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
            #if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
            #    candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
            #else:
            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[replace_pos], topn = 20)
                tmp_words = candidate_words.copy()
                # filter out any low frequency words from candidate list
                for candidate in tmp_words:
                    temp_candidate = candidate[0]
                    if(temp_candidate in self.filtered_embs):
                        if(self.filtered_embs[temp_candidate] < self.filter_min):
                            candidate_words.remove(candidate)

                # make sure some substitutions exist
                if len(candidate_words) == 0:
                    candidate_words = tmp_words.copy()

            except: #if replacement word does not exist in the vocabulary generate random list
                w2v_vocab = list(self.w2v_model.vocab.keys())
                for _ in range(10):
                    rand_pos = random.randint(0, len(w2v_vocab))
                    candidate_words.append((w2v_vocab[rand_pos], 0))

                #tmp_words = candidate_words.copy()
                # filter out any non words from candidate list
                #for candidate in candidate_words:
                #    temp_candidate = candidate[0]
                #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
                #        candidate_words.remove(candidate)

                # if no english words, use anyways
                #if len(candidate_words) == 0:
                #    candidate_words = tmp_words.copy()
        
                
        


            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}

            orig_emb = self.model.embed_sentence(query)
            #print(candidates_words)

            # choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand
            
                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                if(self.detector):
                    obf_pred, obf_prob = self.detector.predict(cur_obf)
                else:
                    obf_pred, obf_prob, _ = self.masker.predict(cur_obf)

                cands_probs[can_word] = obf_prob

                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                # BLSTM returns 0 or 1
                if(self.detector):
                    label = 'OFF'
                else:
                    label = 1

                if(l1_dist < lowest_distance and obf_pred != label):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                    

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest OFF score
            if(not found_replace):
                replace_word = sorted(cands_probs, key = cands_probs.get)[0]
                finished = False


            obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            #print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos], 'l1 dist:', cands_dists[replace_word])

            split_query = obf_query.split()

        return obf_query



    
    # determines word to be chosen via greedy select (checking probability changes) and replaces with explitive letters (!#@*%)
    def GS_PR(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations
        orig_pred, var_probs = self.detector.predictMultiple(variations)
            
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)

            
        replace_pos = prob_diffs.index(max(prob_diffs))
        
        punctAvail = ['!', '#', '@', '*', '%', '$']

        # replace word with combination of punctuation characters
        replace_word = split_query[replace_pos][0] + ''.join([random.choice(punctAvail) for _ in range(len(split_query[replace_pos]) - 1)])

        
        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

        print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])
        return obf_query

        


    def obfuscate(self, query):
        if(self.obf_method == 'GS_GR'):
            return self.GS_GR(query)
        elif(self.obf_method == 'GS_EC'):
            return self.GS_EC(query)
        elif(self.obf_method == 'GS_EC_MAX'):
            return self.GS_EC_MAX(query)
        elif(self.obf_method == 'GS_EC_MAX_Single'):
            return self.GS_EC_MAX_Single(query)
        elif(self.obf_method == 'GS_EC_MAX_Extended'):
            return self.GS_EC_MAX_Extended(query)
        elif(self.obf_method == 'GSS_BF_MAX'):
            return self.GSS_BF_MAX(query)
        elif(self.obf_method == 'GSS_EC_MAX'):
            return self.GSS_EC_MAX(query)
        elif(self.obf_method == 'GS_PR'):
            return self.GS_PR(query)
        elif(self.obf_method == 'AT_EC_MAX'):
            return self.AT_EC_MAX(query)
        

# Walk through a file where each line is text which should be obfuscated.
def main(conversion_file, detector_name, obf_alg = 'GS_GR', embedding_loc = 'glove/glove.twitter.27B.100d.txt', filter_min = 3, filtered_embs = None, alg_version = 0, persp_key = 1):
    
    all_tweets = {}

    # load in text to be obfuscated
    with open(conversion_file, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            all_tweets[tweet[0]] = tweet

    if(filtered_embs == 'None'):
        filtered_embs = None

    # walk through tweets and obfuscate 
    obfuscator = Obfuscator(obf_method = obf_alg, detection_alg=detector_name, embedding_file = embedding_loc, filter_min = filter_min, filtered_embs = filtered_embs, alg_version = alg_version, persp_key = persp_key)

    out_beg = '-'.join(embedding_loc.split('/')[-1].split('.')[:-1])
    if(obf_alg == 'AT_EC_MAX' and detector_name == 'None'):
        detector_name = 'BLSTM'

    if(filtered_embs):
        out_file = open(out_beg + '_obfuscatedVs-' + detector_name + '_' + 'filtered-' + alg_version + '_' + obf_alg + '_' + conversion_file, 'w')
    else:
        out_file = open(out_beg + '_obfuscatedVs-' + detector_name + '_' + obf_alg + '_' + conversion_file, 'w')
    outCSV = csv.writer(out_file, delimiter = '\t')
    for tweet in all_tweets:
        all_tweets[tweet][1] = obfuscator.obfuscate(all_tweets[tweet][1])

        out_tweet = all_tweets[tweet]
        
        outCSV.writerow(out_tweet)




if __name__ == "__main__":
    if(len(sys.argv) == 3):
        main(sys.argv[1], sys.argv[2])
    elif(len(sys.argv) == 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif(len(sys.argv) == 5):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif(len(sys.argv) == 7):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif(len(sys.argv) == 8):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
        
            
