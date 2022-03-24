# Perspective system to determine if a comment is offensive or not based on its toxicity levels
import sys
import json
import requests
import csv
import time 
import string

# a Perspective class object which allows query tests
class Perspective:

    def __init__(self, threshold = 0.5, select = 1):
        self.threshold = threshold
        
        if(int(select) == 1):
            api_key = ''
        elif(int(select) == 2):
            api_key = ''
        #elif(int(select) == 3):
            
        self.url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
                      '?key=' + api_key)


    def predict(self, test_query):
        text = test_query.lower().strip()
        request = createRequestDict(text)
        
        # sleep 2 second to comply with perspective api
        time.sleep(2)

        done = False

        while(not done):
        # send request to google perspective api
            done = True
            try:
                response = requests.post(url=self.url, data=json.dumps(request))
            except:
                done = False
                time.sleep(10)


        response_dict = json.loads(response.content)
        
        if('attributeScores' in response_dict):
            toxicity_level = response_dict['attributeScores']['TOXICITY']['summaryScore']['value']
        else:
            toxicity_level = 0.0

        if(toxicity_level > self.threshold):
            prediction = 'OFF'
        else:
            prediction = 'NOT'

        return prediction, toxicity_level
        

    # allow multiple queries in form of list
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


# create a dicitonary to be sent to the Google perspective api
def createRequestDict(text):
    data_dict = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}
    }
    
    return data_dict
    
# Predicts a comment as Offensive if toxicity is greater than threshold
def PerspectivePredict(testFile, threshold = 0.5, outfile = None):
    api_key = ''
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
                      '?key=' + api_key)

    threshold = float(threshold)

    if(outfile):
        output = open(outfile, 'w')
    else:
        output = open('perspectivePredictionsOut', 'w')
    
    # walk through test file and predict offensive or not
    with open(testFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            text = tweet[1].lower().strip()
            #text = convertChars(text)

            request = createRequestDict(text)
            
            # sleep 2 seconds to comply with perspective api 
            time.sleep(2)
            # send request to google perspective api
            response = requests.post(url=url, data=json.dumps(request))
            response_dict = json.loads(response.content)
            
            if('attributeScores' in response_dict):
                toxicity_level = response_dict['attributeScores']['TOXICITY']['summaryScore']['value']
            else:
                toxicity_level = 0.0

            if(toxicity_level > threshold):
                prediction = 'OFF'
            else:
                prediction = 'NOT'

            output.write(tweet[0] + ',' + prediction + '\n')
        
        output.close()

if(__name__ == "__main__"):
    PerspectivePredict(sys.argv[1], sys.argv[2], sys.argv[3])
