import json
from utils import *
from naive_bayes import NaiveBayes
from calculate_prob import *
import pickle

if __name__ == '__main__':

    test_dir='./test/'   #test folder
    dir_spam='./spam_classifier/p_word_given_spam' # Please mention the root directory of the spam_classifier folder  if the current directory does not contain the spam_classifier folder
    dir_ham='./spam_classifier/p_word_given_ham' # Please mention the root directory of the spam_classifier folder  if the current directory does not contain the spam_classifier folder
    dir_class='./spam_classifier//p_class'  # Please mention the root directory of the spam_classifier folder  if the current directory does not contain the spam_classifier folder



    with open(dir_spam,'rb') as handle:
        p_word_given_spam=pickle.load(handle)
    with open(dir_ham,'rb') as handle:
        p_word_given_ham=pickle.load(handle)
    with open(dir_class,'rb') as handle:
        p_class=pickle.load(handle)
    p_spam=p_class['spam']
    p_ham=p_class['ham']
    

    test_paths=get_data_paths(test_dir)

    
    predictions=[]
    for path in test_paths:
        p_spam_given_input = compute_p_class_given_input(path, p_word_given_spam, p_spam)        
        p_ham_given_input = compute_p_class_given_input(path,p_word_given_ham, p_ham)
        
        if p_spam_given_input > p_ham_given_input:
            predictions.append(1)
        else:
            predictions.append(0)
    
    print('Final predicitons:',predictions)
    
    with open("final_predictions.txt", "w") as output:
        output.write(str(predictions))
        
