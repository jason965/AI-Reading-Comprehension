import tensorflow as tf
import time
import logging
import os
import sys

import numpy as np
from tqdm import tqdm

from data_batcher import get_batch_generator
from vocab import get_glove, get_char_embed
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from modules import RNNEncoder, Bidaf, SelfAttn, SimpleSoftmaxLayer

from tensorflow.python import debug as tf_debug
# Location of pretrained LM.  Here we use the test fixtures.
datadir = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo"
vocab_file = os.path.join(datadir, 'elmo_vocab.txt')
options_file = os.path.join(datadir, 'elmo.json')
weight_file = os.path.join(datadir, 'lm_weight.hdf5')


#Define statistics and hyperparameters
batch_size = 2
hidden_size = 7
context_len = 100
question_len = 25
embedding_size = 50
char_size = 20
num_of_char = 72
max_word_len = 20
dropout = 0.2

#Define path
train_context_path =  "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/train.context"
train_qn_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/train.question"
train_ans_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/train.span"
dev_qn_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/dev.question"
dev_context_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/dev.context"
dev_ans_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/dev.span"
glove_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/glove.6B.50d.txt"


# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

#Load glove
emb_matrix, word2id, id2word = get_glove(glove_path, embedding_size)
#char2id, id2char = get_char_embed()

# Input placeholders
context_elmo = tf.placeholder('int32', shape=(None, None, 50))
question_elmo = tf.placeholder('int32', shape=(None, None, 50))

context_ids = tf.placeholder(tf.int32)
context_mask = tf.placeholder(tf.int32)
qn_ids = tf.placeholder(tf.int32)
qn_mask = tf.placeholder(tf.int32)
ans_span = tf.placeholder(tf.int32, shape=[None, 2])

keep_prob = tf.placeholder_with_default(1.0, shape=())
#---------------------------------------------------------------------
def build_graph():
    def bilm_build_graph(options_file, weight_file):
        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(options_file, weight_file)

        # Get ops to compute the LM embeddings.
        context_embeddings_op = bilm(context_elmo)
        question_embeddings_op = bilm(question_elmo)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        # Our SQuAD model includes ELMo at both the input and output layers
        # of the task GRU, so we need 4x ELMo representations for the question
        # and context at each of the input and output.
        # We use the same ELMo weights for both the question and context
        # at each of the input and output.
        elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)['weighted_op']
        with tf.variable_scope('', reuse=True):
            # the reuse=True scope reuses weights from the context for the question
            elmo_question_input = weight_layers(
                'input', question_embeddings_op, l2_coef=0.0
            )['weighted_op']
        """
        elmo_context_output = weight_layers(
            'output', context_embeddings_op, l2_coef=0.0
        )['weighted_op']

        with tf.variable_scope('', reuse=True):
            # the reuse=True scope reuses weights from the context for the question
            elmo_question_output = weight_layers(
                'output', question_embeddings_op, l2_coef=0.0
            )

        """
        return elmo_context_input, elmo_question_input

    def add_embedding_layer(emb_matrix):
        with tf.variable_scope("embeddings"):
            #set to constant so its untrainable
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            context_embs = tf.nn.embedding_lookup(embedding_matrix, context_ids)
            qn_embs = tf.nn.embedding_lookup(embedding_matrix, qn_ids)
        return context_embs, qn_embs

    #start the main graph
    context_embs, qn_embs = add_embedding_layer(emb_matrix)
    elmo_context_input, elmo_question_input = bilm_build_graph(options_file, weight_file)

    context_embs_concat = tf.concat([elmo_context_input, context_embs], 2) #(2, context_len, 1074)
    qn_embs_concat = tf.concat([elmo_question_input, qn_embs], 2) #(2, question_len, 1074)

    context_embs_concat.set_shape((None, None, 1074))
    qn_embs_concat.set_shape((None, None, 1074))
    qn_mask.set_shape((None,None))
    context_mask.set_shape((None,None))

    with tf.variable_scope("biLSTM"):
        print("Starting biLSTM...")
        LSTMencoder_context = RNNEncoder(hidden_size, keep_prob=keep_prob, cell_type="lstm", input_size=1074)
        LSTMencoder_question = RNNEncoder(hidden_size, keep_prob=keep_prob, cell_type="lstm", input_size=1074)
        #shared weights
        context_hiddens = LSTMencoder_context.build_graph(context_embs_concat, context_mask, scope="context_question_encoder", reuse=False)
        question_hiddens = LSTMencoder_question.build_graph(qn_embs_concat, qn_mask, scope="context_question_encoder", reuse=True)
    
    with tf.variable_scope("bidaf_layer"):
        print("Starting bidaf...")
        bidaf_object = Bidaf(hidden_size*2, keep_prob)
        b = bidaf_object.build_graph(context_hiddens, question_hiddens, context_mask, qn_mask)
    
    with tf.variable_scope("self_attn_layer"):
        SelfAttn_object = SelfAttn(hidden_size, hidden_size*2, keep_prob, input_size=hidden_size*2)
        M = SelfAttn_object.build_graph(b, context_mask, cell_type="lstm") #(batch_size, context_len, hidden_size*2)
    
    with tf.variable_scope("final_lstm_layer"):
        final_lstm_object = RNNEncoder(hidden_size, keep_prob=keep_prob, cell_type="lstm", input_size=hidden_size*2)
        M_prime = final_lstm_object.build_graph(M, context_mask, scope="final_lstm", reuse=False)

    with tf.variable_scope("StartDist"):
        softmax_layer_start = SimpleSoftmaxLayer()
        logits_start, probdist_start = softmax_layer_start.build_graph(M_prime, context_mask)

    with tf.variable_scope("EndDist"):
        softmax_layer_end = SimpleSoftmaxLayer()
        logits_end, probdist_end = softmax_layer_end.build_graph(M_prime, context_mask)

    return logits_start, probdist_start, logits_end, probdist_end

if __name__ == "__main__":

    logits_start, probdist_start, logits_end, probdist_end = build_graph()

    # run the program
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        for batch in get_batch_generator(word2id, char2id, train_context_path, train_qn_path, train_ans_path, batch_size, context_len, question_len, max_word_len, discard_long=True):

            # Create batches of data.
            input_feed = {}
            input_feed[context_elmo] = batcher.batch_sentences(batch.context_tokens)
            input_feed[question_elmo] = batcher.batch_sentences(batch.qn_tokens)
            input_feed[context_ids] = batch.context_ids
            input_feed[context_mask] = batch.context_mask
            input_feed[qn_ids] = batch.qn_ids
            input_feed[qn_mask] = batch.qn_mask
            input_feed[ans_span] = batch.ans_span
            input_feed[keep_prob] = dropout

            print("first context length: ", len(input_feed[context_elmo][0]) - 2)
            print("second context length: ", len(input_feed[context_elmo][1]) - 2 )
            print("first question length: ", len(input_feed[question_elmo][0]) - 2)
            print("second question length: ", len(input_feed[question_elmo][1]) - 2)

            output_feed = [logits_start, probdist_start, logits_end, probdist_end]
            for i in output_feed:
                print(sess.run(i, input_feed).shape)


            break