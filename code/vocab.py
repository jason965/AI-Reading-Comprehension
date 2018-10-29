# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1


def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path
      char: boolean; determine if get glove pretrained character embedding or not 

    Returns:
      emb_matrix: Numpy array shape (400002/96, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print ("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = int(4e5)# this is the vocab size of the corpus we've downloaded

    if 'crawl' in glove_path:
        print("Fastext crawl 300D vocabulary embeddings will be used.")
        vocab_size = 2000000
    elif '840B' in glove_path:
        print("GloVe embeddings of cased 300D will be used.")
        vocab_size = int(21960007)

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        if "crawl" in glove_path:
            fh.readline()
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector) and '840B' not in glove_path:
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            elif glove_dim != len(vector) and '840B' in glove_path:
                print("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
                continue
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

def get_char_embed():
    """
    Create the dicionaries for char2id and id2char
    Note: No char_embedding returned because here its not pretrained but tf.variable to be trained
    
    """
    char_list = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}\""
    char2id = {_PAD:PAD_ID, _UNK:UNK_ID}
    id2char = {PAD_ID:_PAD, UNK_ID:_UNK}
    for (idx,c) in enumerate(char_list):
        idx = idx+2
        char2id.update({c:idx}) 
        id2char.update({idx:c})

    return char2id, id2char

if __name__ == "__main__":
    get_glove(glove_path="/home/lam/squad/data/crawl-300d-2M-subword.vec", glove_dim=300)
