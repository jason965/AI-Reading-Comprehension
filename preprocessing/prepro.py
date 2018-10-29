import argparse
import json
import logging
import nltk
import numpy as np
import random
import os
from tqdm import tqdm


"""
line 100 for no answer question
"""

#set filemode to overwrite previous .log file
logging.basicConfig(filename="prepro_INFO.log", filemode='w', level=logging.INFO, format='%(levelname)s: %(message)s')
random.seed(42)
np.random.seed(42)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    return parser.parse_args()

def tokenize(sequence):
    #note this will include the punctuations
    # Added: remove lowercase()
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return tokens

def total_num_exs(dataset):
    num_exs = 0
    for data in dataset['data']:
        for para in data['paragraphs']:
            num_exs += 1
    return num_exs


def char_word_loc_map(context, context_tokens):
    """
    This function is to check

    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    current_token_idx = 0
    acc = '' #accumulator
    mapping = {}
    for char_idx, char in enumerate(context):
        if char !=" " and char != "\n":
            acc += char
            context_token = str(context_tokens[current_token_idx])
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx) # add to mapping
                acc = '' #rest accumualtor
                current_token_idx += 1
    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(dataset, mode, out_dir):
    num_exs = 0 # number of examples written to file
    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
    examples = []

    for article_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(mode)):
        
        article_paragraphs = dataset['data'][article_id]['paragraphs']
        for context_id in range(len(article_paragraphs)):
            context = str(article_paragraphs[context_id]['context'])
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            #tokenize and lowercase context
            context_tokens = tokenize(context) #Deleted: (alreadly lowercase in tokenize function)
            context = context #Deleted: .lower()

            charloc2wordloc = char_word_loc_map(context, context_tokens)

            #now turn to question
            qas = article_paragraphs[context_id]['qas']

            if charloc2wordloc is None: #there is a problem
                logging.info("Article_id: %d has special characters.", article_id)
                num_mappingprob += len(qas)
                continue #skip this context example

            for qn in qas:
                question_tokens = tokenize(qn['question'])
                question = str(qn['question'])

                #handle question without answer, temporily skip it
                if qn['is_impossible']:
                    continue


                ans_start_charloc = qn['answers'][0]['answer_start']
                #REMOVED: .lower()
                ans_text = str(qn['answers'][0]['text'])
                ans_end_charloc = ans_start_charloc + len(ans_text) #count char (counted one more char)
                
                # Check that the provided character spans match the provided answer text
                if context[ans_start_charloc: ans_end_charloc] != ans_text:
                    logging.info("Article_id: %d, char span and answer text do not match. %s -- %s", article_id, context[ans_start_charloc: ans_end_charloc], ans_text)
                    num_spanalignprob += 1
                    continue

                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1] #e.g ("in", 50)
                ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1]
                assert ans_start_wordloc <= ans_end_wordloc

                # Check retrieved answer tokens match the provided answer text.
                # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                # and the answer character span is around "generation",
                # but the tokenizer regards "fifth-generation" as a single token.
                # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]
                
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    logging.info("Article_id: %d, answer tokens and answer text do not match. %s -- %s", article_id, "".join(ans_tokens), "".join(ans_text.split()))
                    num_tokenprob += 1
                    continue #skip this question/answer pair

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))
                num_exs += 1

    print ("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print ("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print ("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print ("Processed %i examples of total %i\n" % (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))


    #shuffle examples
    indices = np.arange(len(examples))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, mode +'.context'), 'w') as context_file,  \
         open(os.path.join(out_dir, mode +'.question'), 'w') as question_file,\
         open(os.path.join(out_dir, mode +'.answer'), 'w') as ans_text_file, \
         open(os.path.join(out_dir, mode +'.span'), 'w') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]

            # write tokenized data to file
            context_file.write(context+"\n")
            question_file.write(question+"\n")
            ans_text_file.write(answer+"\n")
            span_file.write(answer_span+"\n")

def main():
    args = setup_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"

    #read data
    train_data = json.load(open(train_file,'r'))
    dev_data = json.load(open(dev_file,'r'))

    print("Train data has %d total number of examples." %total_num_exs(train_data))
    print("dev data has %d total number of examples." %total_num_exs(dev_data))

    #preprocess both train and dev data
    preprocess_and_write(train_data, 'train', args.data_dir)
    preprocess_and_write(dev_data, 'dev', args.data_dir)

if __name__ == "__main__":
    main()



                