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