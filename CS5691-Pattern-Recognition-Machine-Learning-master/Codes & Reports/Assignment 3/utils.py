
import nltk
from nltk.tokenize import word_tokenize
import os
import glob
import re
import string   
from nltk.corpus import stopwords   
from nltk.stem import PorterStemmer

def get_words(message):

    #print('m:',message)
    stemmer = PorterStemmer() 
    stopwords_english = stopwords.words('english') 
    message=message.lower()
    message=re.sub('=\\n*\[].:?/','',message)
    tokens=word_tokenize(message)
    #print(tokens)
    clean=[]
    for word in tokens: 
        if (word not in stopwords_english and word not in string.punctuation): 
            clean.append(word)
    stemmed=[]
    for word in clean:
        stem_word = stemmer.stem(word)  
        stemmed.append(stem_word) 
    #print(stemmed) 
    
    return stemmed


def open_file(file_name):

    with open(file_name, 'r') as f:
        try:
            lines = f.read()
        except UnicodeDecodeError:
            print('Unicode decode error [due to undecodable strings], this file is skipped, moving on to next file.')
            lines='a'


    return lines

def get_data_paths(path):

    file_list = glob.glob(path+'*')#os.listdir(path)
    return file_list