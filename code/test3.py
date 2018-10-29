import tensorflow as tf
import time
import logging
import os
import sys

import numpy as np
from tqdm import tqdm

from data_batcher import get_batch_generator
from vocab import get_glove, get_char_embed
from modules import RNNEncoder, masked_softmax, SimpleSoftmaxLayer
from tensorflow.python.ops.rnn_cell import DropoutWrapper

from bilm import Batcher, BidirectionalLanguageModel, weight_layers


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

elmo_dir = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo"
options_file = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo/elmo.json"
weight_file= "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo/lm_weight.hdf5"


class FLAGS(object):
    def __init__(self, batch_size, hidden_size, context_len, question_len, embedding_size, char_size, num_of_char, max_word_len, dropout, elmo_dir):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.context_len = context_len
        self.question_len = question_len
        self.embedding_size = embedding_size
        self.char_size = char_size
        self.num_of_char = num_of_char
        self.max_word_len = max_word_len
        self.dropout = dropout
        self.elmo_dir = elmo_dir

FLAGS = FLAGS(batch_size, hidden_size, context_len, question_len, embedding_size, char_size, num_of_char, max_word_len, dropout, elmo_dir)


glove_path = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squadV2/data/glove.6B.50d.txt"
emb_matrix, word2id, id2word = get_glove(glove_path, FLAGS.embedding_size)
char2id, id2char = get_char_embed()

class QAModel(object):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix, id2char, char2id):
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.emb_matrix =  emb_matrix
        self.id2char = id2char
        self.char2id = char2id
        
        self.batcher = Batcher("/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo/elmo_vocab.txt", 50)
        self.filters = [(5,10)] #change back to 100 after
        
        self.options_file = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo/elmo.json"
        self.weight_file = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo/lm_weight.hdf5"
        
        with tf.variable_scope("QAModel",  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
        self.add_elmo_embedding_layer(self.options_file, self.weight_file)
        with tf.variable_scope("QAModel",  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables("QAModel") #since only one scope "QAModel"
        gradients = tf.gradients(self.loss, params) # d(loss)/d(params) return list of (length len(params))
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0) #return list_clipped, global_norm(here we don't need this)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #This will increment the global step if global_step is not None
        opt = tf.train.AdamOptimizer(learning_rate=0.001) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        self.summaries = tf.summary.merge_all()

    def add_placeholders(self):
        self.context_ids = tf.placeholder(tf.int32)
        self.context_mask = tf.placeholder(tf.int32)
        self.qn_ids = tf.placeholder(tf.int32)
        self.qn_mask = tf.placeholder(tf.int32)
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])
        
        #NOTE:CHANGE
        #self.context_char = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.max_word_len])
        #self.qn_char = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.max_word_len])
        #The following two may not be necessary
        #self.context_char_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.max_word_len])
        #self.qn_char_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.max_word_len])
        self.context_elmo = tf.placeholder('int32', shape=[None, None, 50])
        self.qn_elmo = tf.placeholder('int32', shape=[None, None, 50])
        
        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    def add_embedding_layer(self, emb_matrix):
        with tf.variable_scope("embeddings"):
            #set to constant so its untrainable
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            self.context_embs = tf.nn.embedding_lookup(embedding_matrix, self.context_ids)
            self.qn_embs = tf.nn.embedding_lookup(embedding_matrix, self.qn_ids)

        #self.add_char_embedding_layer()

    def add_elmo_embedding_layer(self, options_file, weight_file, output_use=False):
        """
        Adds ELMo lstm embeddings to the graph.

        Inputs:
            options_file: json_file for the pretrained model
            weight_file: weights hdf5 file for the pretrained model
            output_use: determine if use elmo in output of biRNN (default False)
        """
        #Build biLM graph
        bilm = BidirectionalLanguageModel(options_file, weight_file)
        context_embeddings_op = bilm(self.context_elmo)
        question_embeddings_op = bilm(self.qn_elmo)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        # Our SQuAD model includes ELMo at both the input and output layers
        # of the task GRU, so we need 4x ELMo representations for the question
        # and context at each of the input and output.
        # We use the same ELMo weights for both the question and context
        # at each of the input and output.
        #compute the final ELMo representations.
        self.elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.001)['weighted_op'] #(batch size, context size, ????)
        with tf.variable_scope('', reuse=True):
            # the reuse=True scope reuses weights from the context for the question
            self.elmo_question_input = weight_layers(
                'input', question_embeddings_op, l2_coef=0.001
            )['weighted_op']

        if output_use:
            self.elmo_context_output = weight_layers(
                'output', context_embeddings_op, l2_coef=0.001
            )['weighted_op']
            with tf.variable_scope('', reuse=True):
                # the reuse=True scope reuses weights from the context for the question
                self.elmo_question_output = weight_layers(
                    'output', question_embeddings_op, l2_coef=0.001
                )['weighted_op']
    
    
    def build_graph(self):
        context_embs_concat = tf.concat([self.elmo_context_input, self.context_embs], 2) #(batch_size, qn_len, 1024+self.FLAGS.embedding_size)

        context_embs_concat.set_shape((None, None, 1024+self.FLAGS.embedding_size))
        #qn_embs_concat.set_shape((None, None, 1024+self.FLAGS.embedding_size))
        self.qn_mask.set_shape((None,None))
        self.context_mask.set_shape((None,None))

        with tf.variable_scope("start"):
            softmax_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_start.build_graph(context_embs_concat, self.context_mask)
        with tf.variable_scope("end"):
            softmax_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_end.build_graph(context_embs_concat, self.context_mask)

    def add_loss(self):
        with tf.variable_scope("loss"):
            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start)
            tf.summary.scalar('loss_start', self.loss_start)

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)

    def run_train_iter(self, session, batch, summary_writer):
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        
        #NOTE: CHANGE added context_char
        #input_feed[self.context_char] = batch.context_char
        input_feed[self.context_elmo] = self.batcher.batch_sentences(batch.context_tokens)
        
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        
        #NOTE: CHANGE added qn_char
        #input_feed[self.qn_char] = batch.qn_char
        input_feed[self.qn_elmo] = self.batcher.batch_sentences(batch.qn_tokens)
        
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout
        
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        
        #output_feed = [self.elmo_context_input]
        [_, summaries, loss, global_step, param_norm, gradient_norm] = sess.run(output_feed, feed_dict=input_feed)
        
        print("FINISHED")
    


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        summary_writer = tf.summary.FileWriter("/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad", session.graph)
        for batch in get_batch_generator(self.word2id, self.char2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.question_len, self.FLAGS.max_word_len, discard_long=True):
            self.sample_batch = batch
            
            self.run_train_iter(session, batch, summary_writer)
            break


qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix, id2char, char2id)
#qa_model.add_elmo_embedding_layer(options_file, weight_file)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)
    #variables_names =[v.name for v in tf.trainable_variables()]
    #values = sess.run(variables_names)



