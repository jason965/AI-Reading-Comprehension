import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from data_batcher import get_batch_generator
from evaluate import exact_match_score, f1_score
from pretty_print import print_example
from modules import RNNEncoder, Bidaf, SelfAttn, SimpleSoftmaxLayer
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

logging.basicConfig(level=logging.INFO)

class QAModel(object):
    """
    add_placeholders():
    add_char_embedding_layer
    add_embedding_layer
    build_graph: the main part of the model
    add_loss

    """

    def __init__(self, FLAGS, id2word, word2id, emb_matrix, id2char, char2id):
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.emb =  emb_matrix
        self.id2char = id2char
        self.char2id = char2id


        #This batcher is used for feed_dict in placeholder context_elmo & qn_elmo
        self.batcher = Batcher(os.path.join(self.FLAGS.elmo_dir, "elmo_vocab.txt"), 50)
        self.filters = [(1,124)]

        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix.shape)
        self.add_elmo_embedding_layer(os.path.join(self.FLAGS.elmo_dir, "elmo.json"), os.path.join(self.FLAGS.elmo_dir, "lm_weight.hdf5"))
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables() # also fine tune elmo (original since only one scope "QAModel")
        gradients = tf.gradients(self.loss, params) # d(loss)/d(params) return list of (length len(params))
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm) #return list_clipped, global_norm(here we don't need this)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #This will increment the global step if global_step is not None
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        
        # NOTE: Added exponential moving average for trainable parameters
        #with tf.variable_scope("EMA"):
            #self.ema = tf.train.ExponentialMovingAverage(decay=0.999, zero_debias=True)
            #self.ema_ops = self.ema.apply(tf.trainable_variables())

        #self.ema_saver = tf.train.Saver(max_to_keep=FLAGS.keep)
        #self.ema_bestmodel_saver = tf.train.Saver(max_to_keep=1)
        
        self.summaries = tf.summary.merge_all() #collect all summaries defined in the graph above e.g. in add_loss() function
                                                #This function is like sess.run. The summary writer can then work


    def add_placeholders(self):
        #if shape is not specified, we can pass any shape
        self.context_ids = tf.placeholder(tf.int32)
        self.context_mask = tf.placeholder(tf.int32)
        self.qn_ids = tf.placeholder(tf.int32)
        self.qn_mask = tf.placeholder(tf.int32)
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])
        
        #NOTE:Added char and elmo
        self.context_char = tf.placeholder(tf.int32, shape=[None, None, self.FLAGS.max_word_len])
        self.qn_char = tf.placeholder(tf.int32, shape=[None, None, self.FLAGS.max_word_len])
        self.context_elmo = tf.placeholder(tf.int32, shape=[None, None, 50])
        self.qn_elmo = tf.placeholder(tf.int32, shape=[None, None, 50])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.emb_matrix = tf.placeholder(tf.float32, shape=self.emb.shape)

    def feed_embedding(self,session):
        set_emb=self.embedding_matrix.assign(self.emb_matrix)
        session.run(set_emb,feed_dict={self.emb_matrix:self.emb})

    def add_elmo_embedding_layer(self, options_file, weight_file, output_use=False):
        """
        Adds ELMo lstm embeddings to the graph.
        1. self.elmo_context_input (batch size, max_context_len among the batch, 1024)
        2. self.elmo_question_input (batch size, max_qn_len among the batch, 1024)
        If output_use is true:
            add the output to the graph either

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
        self.elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.001)['weighted_op'] #(batch size, max_context_len among the batch, 1024)
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

        

    def add_char_embedding_layer(self):
    #NOTE: ADD character embeddings
        with tf.variable_scope("embeddings_char"):
            char_embed_matrix = tf.get_variable(name='char_embed_matrix', shape=[self.FLAGS.num_of_char, self.FLAGS.char_size], initializer=tf.initializers.random_uniform(minval=-0.5, maxval=0.5, dtype=tf.float32)) #(71, 20)
            #context_char is placeholder for context char ids
            context_char_emb = tf.nn.embedding_lookup(char_embed_matrix, self.context_char) #shape(batch_size, context_len, max_word_len, char_size)
            #qn_char is placeholder for questionchar ids
            qn_char_emb = tf.nn.embedding_lookup(char_embed_matrix, self.qn_char) #shape(batch_size, question_len, max_word_len, char_size)

            def make_conv(embedding, filters):
                pooled_cnn = []
                for i, (window_size, num_filter) in enumerate(filters):
                    filter_shape = [1, window_size, self.FLAGS.char_size, num_filter]
                    w = tf.get_variable('W_f%s' %i, shape = filter_shape)
                    b = tf.get_variable('b_f%s' %i, shape = [num_filter])
                    conv = tf.nn.conv2d(embedding, filter= w, strides=[1,1,1,1] , padding="VALID") + b #shape(batch_size, context_len, max_word_len-window_size+1, num_filter)
                    conv = tf.nn.relu(conv)
                    
                    h = tf.nn.max_pool(conv, ksize=[1, 1, self.FLAGS.max_word_len - window_size + 1, 1], strides=[1, 1, 1, 1], padding="VALID") #(batch_size, context_len, 1, num_filter)
                    h = tf.squeeze(h, axis=2) # shape (batch_size, context_len, num_filter)
                    pooled_cnn.append(h) 

                return tf.concat(pooled_cnn, axis= 2) #shape (batch_size, context_len, sum all num_filter)

            self.context_char_embs = make_conv(context_char_emb, self.filters) #shape (batch_size, context_len, sum all num_filter)
            #question and context char uses the same embeddings
            tf.get_variable_scope().reuse_variables()
            self.qn_char_embs = make_conv(qn_char_emb, self.filters)  #shape (batch_size, context_len, sum all num_filter)

    def add_embedding_layer(self, emb_matrix_shape):
        with tf.variable_scope("embeddings"):
            with tf.device('/cpu:0'):
                #set to constant so its untrainable
                #embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)
                self.embedding_matrix=tf.Variable(tf.zeros(emb_matrix_shape),trainable=False,name="embedding")
                # Get the word embeddings for the context and question,
                self.context_embs = tf.nn.embedding_lookup(self.embedding_matrix, self.context_ids) #(batch_size, context_len, glove_dim)
                self.qn_embs = tf.nn.embedding_lookup(self.embedding_matrix, self.qn_ids) #(batch_size, qn_len, glove_dim)

        #self.add_char_embedding_layer()

    def build_graph(self):
        """
        Builds the main part of the graph for the model
        
         Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # NOTE CHANGE: concantanate glove and elmo embedding
        # How to handle elmo context_len and glove context_len mismatch?
        # Just make the context_ids no max context_len
        context_embs_concat = tf.concat([self.elmo_context_input, self.context_embs], 2) #(batch_size, qn_len, 1024+self.FLAGS.embedding_size)
        qn_embs_concat = tf.concat([self.elmo_question_input, self.qn_embs], 2) #(batch_size, qn_len, 1024+self.FLAGS.embedding_size)
        
        #set shape so that it can pass to dynamic lstm
        context_embs_concat.set_shape((None, None, 1024+self.FLAGS.embedding_size))
        qn_embs_concat.set_shape((None, None, 1024+self.FLAGS.embedding_size))
        self.qn_mask.set_shape((None,None))
        self.context_mask.set_shape((None,None))

        with tf.variable_scope("biLSTM"):
            Encoder = RNNEncoder(self.FLAGS.hidden_size, keep_prob=self.keep_prob, cell_type="lstm", input_size=1024+self.FLAGS.embedding_size)            
            #shared weights (same scope)
            context_hiddens = Encoder.build_graph(context_embs_concat, self.context_mask, scope="context_question_encoder") #(batch_size, context_len, hidden_size*2)
            question_hiddens = Encoder.build_graph(qn_embs_concat, self.qn_mask, scope="context_question_encoder") #(batch_size, question_len, hidden_size*2)

        with tf.variable_scope("bidaf"):
            bidaf_object = Bidaf(self.FLAGS.hidden_size*2, self.keep_prob)
            b = bidaf_object.build_graph(context_hiddens, question_hiddens, self.context_mask, self.qn_mask) #(batch_size, context_len, hidden_size*8)
        
        with tf.variable_scope("self_attn_layer"):
            SelfAttn_object = SelfAttn(self.FLAGS.hidden_size, self.FLAGS.hidden_size*2, self.keep_prob, input_size=self.FLAGS.hidden_size*2)
            M = SelfAttn_object.build_graph(b, self.context_mask, cell_type="lstm") #(batch_size, context_len, hidden_size*2)
        
        #Make prediction
        with tf.variable_scope('prediction_layer'):
            #Encode the self-attended context first
            with tf.variable_scope("final_lstm_layer"):
                final_lstm_object = RNNEncoder(self.FLAGS.hidden_size, keep_prob=self.keep_prob, cell_type="lstm", input_size=self.FLAGS.hidden_size*2)
                M_prime = final_lstm_object.build_graph(M, self.context_mask, scope="final_lstm") #(batch_size, context_len, h*2)

            #Get start distribution
            with tf.variable_scope("StartDist"):
                softmax_layer_start = SimpleSoftmaxLayer()
                self.logits_start, self.probdist_start = softmax_layer_start.build_graph(M_prime, self.context_mask) #both are (batch_size, context_len)

            with tf.variable_scope("EndDist"):
                logit_start_expand = tf.expand_dims(self.logits_start, axis=2) #(batch_size, context_len, 1)
                blended_end_rnn_input = tf.concat([logit_start_expand, M_prime], axis=2) #(batch_size, context_len, hidden_size*2)
                end_dist_rnn = RNNEncoder(self.FLAGS.hidden_size, keep_prob=self.keep_prob, direction="unidirectional")
                end_rnn_output = end_dist_rnn.build_graph(blended_end_rnn_input, self.context_mask, scope="end_dist_rnn")

                # Get the end dist
                softmax_layer_end = SimpleSoftmaxLayer()
                self.logits_end, self.probdist_end = softmax_layer_end.build_graph(end_rnn_output, self.context_mask)
            

    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """

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
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.context_elmo] = self.batcher.batch_sentences(batch.context_tokens)
        #NOTE: CHANGE added context_char
        #input_feed[self.context_char] = batch.context_char
        
        
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.qn_elmo] = self.batcher.batch_sentences(batch.qn_tokens)
        #NOTE: CHANGE added qn_char
        #input_feed[self.qn_char] = batch.qn_char
        

        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)


        return loss, global_step, param_norm, gradient_norm

    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask

        padded_batch_context_tokens = []
        for context in batch.context_tokens:
            if len(context) > self.FLAGS.context_len:
                padded_batch_context_tokens.append(context[:self.FLAGS.context_len])
            else:
                padded_batch_context_tokens.append(context)
        input_feed[self.context_elmo] = self.batcher.batch_sentences(padded_batch_context_tokens)

        #NOTE: CHANGE added context_char
        #input_feed[self.context_char] = batch.context_char
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask

        padded_batch_qn_tokens = []
        for qn in batch.qn_tokens:
            if len(qn) > self.FLAGS.question_len:
                padded_batch_qn_tokens.append(qn[:self.FLAGS.question_len])
            else:
                padded_batch_qn_tokens.append(qn)
        input_feed[self.qn_elmo] = self.batcher.batch_sentences(padded_batch_qn_tokens)

        #NOTE: CHANGE added qn_char
        #input_feed[self.qn_char] = batch.qn_char
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss

    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.

        for batch in get_batch_generator(self.word2id, self.char2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.question_len, self.FLAGS.max_word_len, discard_long=True):
            # Get loass for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size) #mutiply by curr_batch_size since the loss is divided by this already
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print ("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic))

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)
        return dev_loss

    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        padded_batch_context_tokens = []
        for context in batch.context_tokens:
            if len(context) > self.FLAGS.context_len:
                padded_batch_context_tokens.append(context[:self.FLAGS.context_len])
            else:
                padded_batch_context_tokens.append(context)
        input_feed[self.context_elmo] = self.batcher.batch_sentences(padded_batch_context_tokens)
        #input_feed[self.context_elmo] = self.batcher.batch_sentences(batch.context_tokens)
        
        #NOTE: CHANGE added context_char
        #input_feed[self.context_char] = batch.context_char

        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        padded_batch_qn_tokens = []
        for qn in batch.qn_tokens:
            if len(qn) > self.FLAGS.question_len:
                padded_batch_qn_tokens.append(qn[:self.FLAGS.question_len])
            else:
                padded_batch_qn_tokens.append(qn)
        input_feed[self.qn_elmo] = self.batcher.batch_sentences(padded_batch_qn_tokens)
        #input_feed[self.qn_elmo] = self.batcher.batch_sentences(batch.qn_tokens)
        #NOTE: CHANGE added qn_char
        #input_feed[self.qn_char] = batch.qn_char

        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end] #defined in the end of build_graph()
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch, span="dp"):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        start_dist, end_dist = self.get_prob_dists(session, batch) #numpy array

        if span == "original":
            start_pos = np.argmax(start_dist, axis=1)
            end_pos = np.argmax(end_dist, axis=1)
        elif span == "dp":
            """
            The thoery is add more weights to early start_pos context
            Then for the words equal or after that start_pos, find the word that has max dist
            The words in between (inclusive) are the answer
            """

            end_dp=np.zeros(end_dist.shape)
            end_dp[:,-1]=end_dist[:,-1]
            for i in range(len(end_dist[0])-2,-1,-1):
                end_dp[:,i]=np.amax([end_dist[:,i],end_dp[:,i+1]],axis=0)
            start_pos=np.argmax(start_dist*end_dp,axis=1)
            end_pos=map(lambda i:start_pos[i]+np.argmax(end_dist[i,start_pos[i]:]),range(len(end_dist)))

        #print(start_dist)
        #print(start_dist.shape)
        #print(end_dist)
        #print(end_dist.shape)
        return start_pos, end_pos


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." %(str(num_samples) if num_samples!= 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, self.char2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.question_len, self.FLAGS.max_word_len, discard_long=False):
            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch) #numpy arrays shape (batch_size)

            # Convert the start and end positions to lists (length batch_size)
            try:
                pred_start_pos = pred_start_pos.tolist()
                pred_end_pos = pred_end_pos.tolist()
            except:
                pred_start_pos = [pos for pos in pred_start_pos]
                pred_end_pos = [pos for pos in pred_end_pos]

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM              
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start: pred_ans_end+1]
                pred_answer = " ".join(pred_ans_tokens)

                #Get the true answer (No UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calculate F1, EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                # Either all or only num_samplse for the calculation
                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))
        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.
        Note: all the inputs of this function are defined in main.py
        This function will be run in main.py

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        
        #ema_checkpoint_path = os.path.join(self.FLAGS.train_dir, "ema_qa.ckpt")
        #ema_bestmodel_dir = os.path.join(self.FLAGS.train_dir, "ema_best_checkpoint")
        #ema_bestmodel_ckpt_path = os.path.join(ema_bestmodel_dir, "qa_best.ckpt")
        
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0
        logging.info("Beginning training loop...")
        #Note if self.FLAGS.num_epochs == 0, then train infinitely
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, self.char2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.question_len, self.FLAGS.max_word_len, discard_long=True):
                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)
                    
                    # NOTE: CHANGE
                    #logging.info("Saving to %s..." % ema_checkpoint_path)
                    #self.ema_saver.save(session, ema_checkpoint_path, global_step=global_step)
                    
                
                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)
                        
                        # NOTE: CHANGE
                        #logging.info("Saving to %s..." % ema_bestmodel_ckpt_path)
                        #self.ema_bestmodel_saver.save(session, ema_bestmodel_ckpt_path, global_step=global_step)
                        

            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()


#used in train() function
def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
