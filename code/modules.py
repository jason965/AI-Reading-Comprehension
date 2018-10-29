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

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class SelfAttn(object):

    def __init__(self, hidden_size, value_vec_size, keep_prob, input_size):
        self.hidden_size = hidden_size
        self.value_vec_size = value_vec_size
        self.keep_prob = keep_prob
        self.input_size = input_size

    def build_graph(self, inputs, mask, cell_type):
        """
        The attention uses concatenation and dot product but not the original Rnet as it has huge memory
        consumption.

        inputs: output from Bidaf #(batch_size, context_len, value_vec_size) here value_vec_size ==hidden_size*2
        mask: context_mask #(batch_size, context_len)
        """
        with tf.variable_scope("selfattn_lstm"):
            SelfAttn_lstm_encoder = RNNEncoder(self.hidden_size, self.keep_prob, cell_type, input_size=self.input_size)
            SelfAttn_hiddens = SelfAttn_lstm_encoder.build_graph(inputs, mask, scope="SelfAttn_BILSTM") # (batch_size, context_len, hidden_size*2)

        #construct similarity matrix
        with tf.variable_scope("selfattn_similarity_matarix"):
            initializer = tf.contrib.layers.xavier_initializer()
            W_1 = tf.get_variable("W_1", shape=[self.value_vec_size, ], initializer=initializer)
            W_2 = tf.get_variable("W_2", shape=[self.value_vec_size, ], initializer=initializer)
            W_3 = tf.get_variable("W_3", shape=[self.value_vec_size, ], initializer=initializer)

            first = tf.expand_dims(tf.tensordot(SelfAttn_hiddens, W_1, [(2), (0)]), 2) #(batch_size, context, 1)
            second =  tf.expand_dims(tf.tensordot(SelfAttn_hiddens, W_2, [(2), (0)]), 1) #(batch_size, 1, context)

            #handle the elementwise multiplcation term (also need to - large value for diagonal)
            gi_o_gj = tf.multiply(W_3, SelfAttn_hiddens) #(batch_size, context, value_vec_size)
            inputs_transpose = tf.transpose(SelfAttn_hiddens, perm=[0,2,1]) #(batch_size, value_vec_size, context_len)
            third = tf.matmul(gi_o_gj, inputs_transpose) #(batch_size, context_len, context_len)

            concat = first+second+third #(batch_size, context_len, context_len)

            #force diagonal element and padded context to be very small
            concat_mask_diagonal = tf.eye(tf.shape(SelfAttn_hiddens)[1], batch_shape=[tf.shape(SelfAttn_hiddens)[0]]) * int(-1e30) 
            concat_mask_padded = tf.expand_dims(mask, axis=1) * tf.expand_dims(mask, axis=2) #(batch_size, 1, context_len) * (batch_size, context_len, 1) -> (batch_size, context_len, context_len)
            concat_mask_padded = (1 - tf.cast(concat_mask_padded, 'float')) * int(-1e30) #(batch_size, context_len, context_len)
            self_attn = concat + concat_mask_diagonal + concat_mask_padded #(batch_size, context_len, context_len)

            # Allow zero-attention by adding a learned bias to the normalizer. Apply softmax.
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            self_attn = tf.exp(self_attn)
            self_attn = self_attn / (tf.reduce_sum(self_attn, axis=2, keep_dims=True) + bias)

            #
            m = tf.matmul(self_attn, SelfAttn_hiddens) #(batch_size, context_len, value_vec_size)
            gate_input = tf.concat([m, SelfAttn_hiddens, m*SelfAttn_hiddens], axis=2) #(batch_size, context_len, value_vec_size*3)
            attn_gate = tf.contrib.layers.fully_connected(gate_input, num_outputs=self.value_vec_size, activation_fn=tf.nn.relu) #(batch_size, context_len, value_vec_size)

        out = inputs + attn_gate #(batch_size, context_len, value_vec_size)
        out = tf.nn.dropout(out, self.keep_prob)
        return out

class Bidaf(object):

    def __init__(self, value_vec_size, keep_prob):
        #here value_vec_size == hidden_size*2
        #self.qn_len = qn_len
        self.value_vec_size = value_vec_size
        self.keep_prob = keep_prob
    def build_graph(self, context_hiddens, question_hiddens, context_mask, qn_mask):
        initializer = tf.contrib.layers.xavier_initializer()
        W_ht = tf.get_variable("W_ht", shape=[self.value_vec_size, ], initializer=initializer)
        W_uj = tf.get_variable("W_uj", shape=[self.value_vec_size, ], initializer=initializer)
        W_hou = tf.get_variable("W_hou", shape=[self.value_vec_size, ], initializer=initializer)

        S_ht = tf.expand_dims(tf.tensordot(context_hiddens, W_ht, [(2), (0)]), 2) # (batch_size, context_len) -> (batch_size, context_len, 1)
        S_uj = tf.expand_dims(tf.tensordot(question_hiddens, W_uj, [(2), (0)]), 1) # (batch_size, qn_len) -> (batch_size, 1, qn_len)

        #first get WT * ht. Then get the multiplicative dot product wT dot (ht * uj)
        woh = tf.multiply(W_hou, context_hiddens) #(batch_size, context_len, hidden_size*2)
        qn_transpose = tf.transpose(question_hiddens, perm=[0,2,1]) #(batch_size, hidden_size*2, qn_len)
        S_woh = tf.matmul(woh, qn_transpose) #(batch_size, context_len, qn_len)

        S = S_ht + S_uj + S_woh #(batch_size, context_len, qn_len)

        #C2Q, the new mask also masked the padded context so the padded context has same weight for each qn words
        bidaf_C2Q_mask = tf.expand_dims(context_mask, -1) * tf.expand_dims(qn_mask, 1) #(batch_size, context_len, 1) * (batch_size, 1, qn_len)
        _ , qn_attn = masked_softmax(S, bidaf_C2Q_mask, 2) #(softmax in qn_len axis) shape: (batch_size, context_len, qn_len)
        C2Q = tf.matmul(qn_attn, question_hiddens) #(batch_size, context_len, hidden_size*2)

        #Q2C
        row_max = tf.reduce_max(S, axis= 2) #(batch_size, context_len)
        _ , context_attn = masked_softmax(row_max, context_mask, 1) #(batch_size, context_len)

        context_attn  = tf.expand_dims(context_attn, 1)  #(batch_size, 1, context_len)
        Q2C = tf.matmul(context_attn, context_hiddens) #(batch_size, 1, hidden_size*2)

        #concat Q2C, C2Q & context_hiddens
        b_concat4 = tf.multiply(context_hiddens, Q2C) #(batch_size, context_len, hidden_size*2)
        b_concat3 = tf.multiply(context_hiddens, C2Q) #(batch_size, context_len, hidden_size*2)
        output = tf.concat([context_hiddens, C2Q, b_concat3, b_concat4], axis=2)
        #map it back to hidden_size *2 (can remove this line, i think its not particularly helpful)
        output = tf.contrib.layers.fully_connected(output, num_outputs=self.value_vec_size, activation_fn=tf.nn.relu)
        output = tf.nn.dropout(output, self.keep_prob)  #(batch_size, context_len, hidden_size*8)
        return output



class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.

    NOTE: this code cannot use cudnn (i don't know how to do with variable length!)
    """

    def __init__(self, hidden_size, keep_prob, cell_type="rnn", use_cudnn=False, direction="bidirectional", input_size=None):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
          cell_type: str, either gru or lstm
          
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.use_cudnn = use_cudnn
        self.direction = direction

        if self.use_cudnn:
            #print("Using cudnn for bidirectional %s" %cell_type)
            if cell_type == "gru":
                self.rnn_cell_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=self.hidden_size, direction=self.direction)
                self.rnn_cell_bw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=self.hidden_size, direction=self.direction)
            elif cell_type == "lstm":
                self.rnn_cell_fw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.hidden_size, direction=self.direction)
                self.rnn_cell_bw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.hidden_size, direction=self.direction)

        else:
            #print("cudnn is not used for bidirectional %s" %cell_type)
            if cell_type == "gru":
                self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
                if self.direction == "bidirectional":
                    self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
            elif cell_type == "lstm":
                self.rnn_cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
                if self.direction == "bidirectional":
                    self.rnn_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)

            elif cell_type == "rnn":
                self.rnn_cell_fw = tf.contrib.rnn.BasicRNNCell(self.hidden_size, activation=tf.nn.relu)
                if self.direction == "bidirectional":
                    self.rnn_cell_bw = tf.contrib.rnn.BasicRNNCell(self.hidden_size, activation=tf.nn.relu)
            else:
                raise Exception("wrong cell_type gru/lstm")

        #DropoutWrapper
        if input_size is not None: #variational dropout
            self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob , state_keep_prob=self.keep_prob, variational_recurrent=True, input_size=input_size , dtype=tf.float32)
            if self.direction == "bidirectional":
                self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob, state_keep_prob=self.keep_prob, variational_recurrent=True, input_size=input_size, dtype=tf.float32)
        else:
            self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            if self.direction == "bidirectional":
                self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)


    def build_graph(self, inputs, masks, scope=None):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        if scope is None:
            scope = "RNNEncoder"

        with tf.variable_scope(scope):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            if self.direction == "bidirectional":
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
                # Concatenate the forward and backward hidden states
                out = tf.concat([fw_out, bw_out], 2)
            elif self.direction == "unidirectional":
                out, _ = tf.nn.dynamic_rnn(self.rnn_cell_fw, inputs, input_lens, dtype=tf.float32)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.
    Do not want the prediction is on the padded words (which do not exist)

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
