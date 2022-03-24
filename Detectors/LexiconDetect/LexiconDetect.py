# Lexicon based system to determine if a comment is offensive or not based on a lexicon of offensive words
import sys
import csv 
import emoji
import string
import wordsegment
wordsegment.load()


# a LexiconDetect object which allow query testing
class LexiconDetect:
    
    def __init__(self, offensive_word_file = 'abusive_words.txt'):
        # load in offensive words
        self.offensive_words = set()
        offensive_file = open(offensive_word_file)
        
        for line in offensive_file:
            self.offensive_words.add(line.strip())
        
        offensive_file.close()
        

    # Predicts a comment as Offensive if post contains an abusive word
    def predict(self, test_query):
        text = preProcessText(test_query).split()
        
        is_offensive = False
            
        for cur_word in text:
            if(cur_word in self.offensive_words):
                is_offensive = True
        
        if(is_offensive):
            prediction = 'OFF'
            prob = 1
        else:
            prediction = 'NOT'
            prob = 0
        
        return prediction, prob


    # allows multiple queries at once 
    def predictMultiple(self, test_queries):
        predictions = []
        probs = []
        for test in test_queries:
            pred, prob = self.predict(test)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs

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


def preProcessText(text):
    text = text.lower().strip()
    #text = convertChars(text)
    
    # change emojis to readable text
    # segment text
    text = ' '.join(wordsegment.segment(emoji.demojize(text)))

    return text
    
# Predicts a comment as Offensive if post contains an abusive word
def LexiconPredict(testFile, offensive_word_file = 'abusive_words.txt', outfile = None):

    # load in offensive words
    offensive_words = set()
    offensive_file = open(offensive_word_file)
    for line in offensive_file:
        offensive_words.add(line.strip())

    offensive_file.close()

    if(outfile):
        output = open(outfile, 'w')
    else:
        output = open('lexiconPredictionsOut', 'w')
    
    # walk through test file and predict offensive or not
    with open(testFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            text = preProcessText(tweet[1]).split()
            is_offensive = False
            
            for cur_word in text:
                if(cur_word in offensive_words):
                    is_offensive = True
            
            if(is_offensive):
                prediction = 'OFF'
            else:
                prediction = 'NOT'

            output.write(tweet[0] + ',' + prediction + '\n')
        
        output.close()


if(__name__ == "__main__"):
    LexiconPredict(sys.argv[1], sys.argv[2], sys.argv[3])
