#Initializing all the required libraries 
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


#Reading the file that needs to be extract into summary
def read_textfile(file_name):
    sentences = []
    file = open(file_name, 'r') 
    text_data = file.readlines()
    text_data = [x for x in text_data if x != '\n'] # it should remove any break present
    text_data = [x.replace('\n',' ') for x in text_data] #this would remove that end of line
    text_data = ''.join(text_data) 
    #splitiing the data
    text = text_data.split('. ') 
    for sentence in text:
        sentences.append(sentence.replace("^[a-zA-Z0-9!@#$&()-`+,/\"]", " ").split(" ")) 
    return sentences

#Finding similarities in the sentences and stopwords such as the etc....
def sentence_similarity(sentence1, sentence2, stop_words=None):
    if stop_words is None:
        stop_words = [] #stopwords will be added on the empty array
 
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]
 
    all_words = list(set(sentence1 + sentence2)) #list will have all the words from sent1 and sent2
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 