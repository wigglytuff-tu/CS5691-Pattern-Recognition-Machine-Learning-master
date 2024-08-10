

from calculate_prob import *
import pickle


class NaiveBayes:
    def __init__(self, vocab_size):
  
        self.p_word_given_spam = dict()
        self.p_word_given_ham = dict()
        self.p_spam = 0
        self.p_ham = 0
        self.vocab_size = vocab_size
        

    def train(self, X_paths_spam, X_paths_ham):

        self.p_word_given_spam = compute_p_word_given_class(X_paths_spam, self.vocab_size)
        self.p_word_given_ham = compute_p_word_given_class(X_paths_ham, self.vocab_size)
        self.p_spam = compute_p_class(len(X_paths_spam), len(X_paths_ham))
        self.p_ham = compute_p_class(len(X_paths_ham), len(X_paths_spam))
        ls={}
        ls['spam']=self.p_spam
        ls['ham']=self.p_ham
        with open('p_word_given_spam','wb') as handle:
            pickle.dump(self.p_word_given_spam ,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('p_word_given_ham','wb') as handle:
            pickle.dump(self.p_word_given_ham,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('p_class','wb') as handle:
            pickle.dump(ls ,handle, protocol=pickle.HIGHEST_PROTOCOL)
        


    def predict(self, X_instance_path):
 
        p_spam_given_input = compute_p_class_given_input(X_instance_path, self.p_word_given_spam, self.p_spam)        
        p_ham_given_input = compute_p_class_given_input(X_instance_path, self.p_word_given_ham, self.p_ham)
        #print(p_spam_given_input,p_ham_given_input)
        
        if p_spam_given_input > p_ham_given_input:
            return 1
        else:
            return 0
        
        
    def evaluate(self, X_paths, ground_truth_class):
   
        gt = 1 if ground_truth_class == 'spam' else 0

        count = 0
        for path in X_paths:
            if self.predict(path) == gt:
                count += 1

        return float(count)/len(X_paths)