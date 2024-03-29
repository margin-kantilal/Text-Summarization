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

     # build the vector for the first sentence
    for w in sentence1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
    for w in sentence2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stopwords):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)

    return similarity_matrix


def create_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_textfile(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ".".join(summarize_text))

 
 # let's begin
    create_summary( "msft.txt", 2)

