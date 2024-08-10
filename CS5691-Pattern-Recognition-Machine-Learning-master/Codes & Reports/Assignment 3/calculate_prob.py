from utils import get_words, open_file

import math

def compute_p_word_given_class(data_paths, vocab_size):

    count_words = {}
    total_num_words = 0
    
    for path in data_paths:
        words = get_words(open_file(path))
        total_num_words += len(words)
        for word in words:
            if word in count_words:
                count_words[word] += 1
            else:
                count_words[word] = 1

    p_word_given_class = {}

    for word in count_words:
        prob = (count_words[word] + 1) / (total_num_words + vocab_size + 1)
        p_word_given_class[word] = prob
        #print(word + " ", end="")
        #print(p_word_given_class[word])

    p_word_given_class["UNKNOWN_WORD"] = 1 / (total_num_words + vocab_size + 1)


    return p_word_given_class


def compute_p_class(n_samples_this_class, n_samples_other_class):


    n_samples_this_class += 1
    p_class = n_samples_this_class / (n_samples_this_class + n_samples_other_class + 1)
    return p_class


def compute_p_class_given_input(input_path, p_word_given_class, p_class):
  
    words = get_words(open_file(input_path))
    p_class_given_input = 0

    for word in words:
        prob = p_word_given_class["UNKNOWN_WORD"]
        if word in p_word_given_class:
            prob = p_word_given_class[word]
        p_class_given_input += math.log(prob)

    p_class_given_input += math.log(p_class)
 
    return p_class_given_input