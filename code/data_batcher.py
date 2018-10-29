import random
import time
import numpy as np
import re
from vocab import PAD_ID, UNK_ID


class Batch(object):
    def __init__(self, context_ids_batch, context_mask, context_tokens_batch, context_char_ids_batch, context_char_mask, context_char_tokens_batch, qn_ids_batch, qn_mask, qn_tokens_batch, qn_char_ids_batch, qn_char_mask, qn_char_tokens_batch, ans_span_batch, ans_tokens_batch, uuids=None):
        """
        Inputs:
          {context/qn}_ids_batch: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_char_ids_batch: Numpy arrays.
            Shape (batch_size, {context_len/question_len}, max_word_len). Contains padding.

          {context/qn}_mask: Numpy arrays, same shape as _ids_batch.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn}_char_mask: Numpy arays, same shape as _char_ids_batch.
            Contains 1s where there is real data, 0s where there is padding.

          {context/qn/ans}_tokens_batch: List length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
          {context/qn}_char_tokens: Lists length batch_size

          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids_batch
        self.context_mask = context_mask
        self.context_tokens = context_tokens_batch
        
        #Added char
        self.context_char = context_char_ids_batch
        self.context_char_mask = context_char_mask
        self.context_char_tokens = context_char_tokens_batch

        self.qn_ids = qn_ids_batch
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens_batch
        
        #Added char
        self.qn_char = qn_char_ids_batch
        self.qn_char_mask = qn_char_mask
        self.qn_char_tokens = qn_char_tokens_batch

        self.ans_span = ans_span_batch
        self.ans_tokens = ans_tokens_batch

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)



def split_by_whitespace(sentence):
    words = []
    for word in sentence.strip().split():
        words.extend(re.split(" ", word))
    return [w for w in words if w]

def sentence_to_token_ids(sentence, word2id, char2id=None):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    
    If char2id is not None, then it will return char_ids: a list of lists of character ids
    """
    tokens = split_by_whitespace(sentence) # list of strings
    #Added: if can't find case sensitive words, find its lower case, if both fail, then UNK
    ids = [word2id.get(w, word2id.get(w.lower(), UNK_ID)) for w in tokens]
    char_tokens = [[char for char in word] for word in tokens]
    char_ids = [[char2id.get(c, UNK_ID) for c in w] for w in tokens]
    return tokens, ids, char_tokens, char_ids

def intstr_to_intlist(intstring):
    return [int(s) for s in intstring.split()]


def padded(token_batch, batch_pad=0, word_len=0, char_pad=False):
    """
    Note: map function will map the inner list
    e.g. x = [[1,2,3], [4,5,6]], map(lambda x: len(x), token_batch) -> [3,3]
    The list mapped here [1,2,3] and [4,5,6]
    
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
      char_pad: Boolean. Determine if padding the character tokens or not
      word_len: context_len/question_len. pad the words for character
    Returns:
      List (length batch_size) of padded of lists of ints.
      All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    if char_pad:
        token_batch = list(token_batch) #change from tuple to list since tuple cannot change its element
        #loop through each example
        for i in range(len(token_batch)):
            #pad the words first
            while len(token_batch[i]) < word_len:
                token_batch[i].append(list([PAD_ID]))
            #now pad the characters for each word
            token_batch[i] = padded(token_batch[i], batch_pad, word_len=0, char_pad=False)
        token_batch = tuple(token_batch) #change back to tuple
        return token_batch
    
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return list(map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch))


def refill_batches(batches ,word2id, char2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, max_word_len, discard_long):
    """
     #NOTE: Need to deal with the no answer question ans pair in the future
    Adds more batches into the "batches" list ascendingly.

    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.

    batches:
      {context/qn}_{ids/tokens}_batch: list (length batch_size) of lists of {ids/tokens}
      ans_{span/tokens}_batch: similar
    """
    print("Refilling batches...")
    tic = time.time()
    examples = []

    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()
    while context_line and qn_line and ans_line: #while not reach an end
        context_tokens, context_ids, context_char_tokens, context_char_ids = sentence_to_token_ids(context_line, word2id, char2id)
        qn_tokens, qn_ids, qn_char_tokens, qn_char_ids = sentence_to_token_ids(qn_line, word2id, char2id)
        ans_span = intstr_to_intlist(ans_line) # a list of integers (should be 2 int)

        #read next line 
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        #check if ans_span has two integers and latter one is larger than former one
        #Note that later the ans_span for no answer needs changes
        assert len(ans_span) == 2, "The answer span is incorrect"
        if ans_span[0] > ans_span[1]:
            print ("Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1]))
            continue

        ans_tokens = context_tokens[ans_span[0]: ans_span[1]+1] # list of strings

        #truncate too-long questions then the characters
        if len(qn_char_ids) > question_len:
            qn_char_ids = qn_char_ids[:question_len]
        for i in range(len(qn_char_ids)):
            if len(qn_char_ids[i]) > max_word_len:
                qn_char_ids[i] = qn_char_ids[i][:max_word_len]


        #discard / truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else: #truncate
                qn_ids = qn_ids[:question_len]
        
        #truncate too-long context then the characters
        if len(context_char_ids) > context_len:
            context_char_ids = context_char_ids[:context_len]
        for i in range(len(context_char_ids)):
            if len(context_char_ids[i]) > max_word_len:
                context_char_ids[i] = context_char_ids[i][:max_word_len]

        #discard / truncate too-long questions
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else:
                context_ids = context_ids[:context_len]
                

        #if everything is good, then add to examples
        examples.append((context_ids, context_tokens, context_char_tokens, context_char_ids, qn_ids, qn_tokens, qn_char_tokens, qn_char_ids, ans_span, ans_tokens))

        #stop reflling if enough batches -- set to 160 batches
        if len(examples) == batch_size * 160:
            break


    #After reached an end or have 160 batches
    #sort the batches by question (NOT by context since each context has many questions so may repeat the same context many times)
    examples = sorted(examples, key=lambda e: len(e[2]))

    for batch_start in range(0, len(examples), batch_size):
        #NOTE: 
        context_ids_batch, context_tokens_batch, context_char_tokens_batch, context_char_ids_batch, qn_ids_batch, qn_tokens_batch, qn_char_tokens_batch, qn_char_ids_batch, ans_span_batch, ans_tokens_batch = zip(*examples[batch_start: batch_start+batch_size])
        #NOTE: zip(*a) e.g. a = [[1,2,3], [4,5,6], [7,8,9]]
        #zip(*a) --> [(1,4,7), (2,5,8), (3,6,9)]        

        batches.append((context_ids_batch, context_tokens_batch, context_char_tokens_batch, context_char_ids_batch, qn_ids_batch, qn_tokens_batch, qn_char_tokens_batch, qn_char_ids_batch, ans_span_batch, ans_tokens_batch))

    random.shuffle(batches)
    toc = time.time()

    print("Refilling batches took %.2f seconds" %(toc-tic))

def get_batch_generator(word2id, char2id, context_path, qn_path, ans_path, batch_size, context_len, question_len, max_word_len, discard_long):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """

    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    batches = []

    while True:
        if len(batches) == 0: #add more batches
            refill_batches(batches ,word2id, char2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, max_word_len, discard_long)  
        if len(batches) == 0: #if after filling still no examples
            break

        #Get a batch and remove this from batches
        (context_ids_batch, context_tokens_batch, context_char_tokens_batch, context_char_ids_batch, qn_ids_batch, qn_tokens_batch, qn_char_tokens_batch, qn_char_ids_batch, ans_span_batch, ans_tokens_batch) = batches.pop(0)

        #turn the below to numpy array (since for model input)
        #pad the context_ids & qn_ids
        context_ids_batch = np.array(padded(context_ids_batch, 0, char_pad=False))
        context_char_ids_batch = np.array(padded(context_char_ids_batch, max_word_len, 0, char_pad=True))
        qn_ids_batch = np.array(padded(qn_ids_batch, 0, char_pad=False))
        qn_char_ids_batch = np.array(padded(qn_char_ids_batch, max_word_len, 0, char_pad=True))

        context_mask = (context_ids_batch != PAD_ID).astype(np.int32)
        qn_mask = (qn_ids_batch != PAD_ID).astype(np.int32)
        context_char_mask = (context_char_ids_batch != PAD_ID).astype(np.int32)
        qn_char_mask = (qn_char_ids_batch != PAD_ID).astype(np.int32)

        # Make ans_span into a np array
        ans_span_batch = np.array(ans_span_batch) # shape (batch_size, 2)

        batch = Batch(context_ids_batch, context_mask, context_tokens_batch, context_char_ids_batch, context_char_mask, context_char_tokens_batch, qn_ids_batch, qn_mask, qn_tokens_batch, qn_char_ids_batch, qn_char_mask, qn_char_tokens_batch, ans_span_batch, ans_tokens_batch, uuids=None)
        """
        print("shape of context_ids: ", context_ids_batch.shape)
        print("shape of context_mask: ", context_mask.shape)
        print("qn_ids shape: ", qn_ids_batch.shape)
        print("qn_mask shape: ", qn_mask.shape)
        """
        yield batch

