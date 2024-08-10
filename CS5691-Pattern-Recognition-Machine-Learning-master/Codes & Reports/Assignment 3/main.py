
import json
from utils import *
from naive_bayes import NaiveBayes


if __name__ == '__main__':

    # data locations
    root = '/home/kamalesh/Documents/enron1/'
    train_path_spam = root + 'spam_train/'
    train_path_ham = root + 'ham_train/'
    test_path_spam = root + 'spam_test/'
    test_path_ham = root + 'ham_test/'


    X_train_spam = get_data_paths(train_path_spam)  
    X_train_ham = get_data_paths(train_path_ham)
    X_test_spam = get_data_paths(test_path_spam) 
    X_test_ham = get_data_paths(test_path_ham) 



    vocab = set()
    for path in X_train_spam + X_train_ham:
        message = open_file(path)
        words = get_words(message)
        vocab = vocab.union(set(words))
    vocab_size = len(vocab)



    nb = NaiveBayes(vocab_size)
    nb.train(X_train_spam, X_train_ham)


    spam_acc = nb.evaluate(X_test_spam, 'spam')
    ham_acc = nb.evaluate(X_test_ham, 'ham')
        
    print('test accuracies:', 'spam emails', '{0:.3f}'.format(spam_acc)+',', \
          'ham emails', '{0:.3f}'.format(ham_acc))
