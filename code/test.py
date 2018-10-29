#import tensorflow as tf
import os
import numpy as np

"""
#The use of np.amax and dynamic programming

a = np.array([1,3,9])
b = np.array([0,5,4])
print(np.amax([a,b], axis=0)) #(1,5,9)
"""
start_dist = np.array([[5,7,2],[0,5,3],[3,8,3],[7,2,9],[6,7,2]])
end_dist = np.array([[4,7,6],[3,8,2],[9,0,8],[3,2,6],[5,5,3]])
end_dp=np.zeros(end_dist.shape)
end_dp[:,-1]=end_dist[:,-1]
print("end_dist[0] length:", len(end_dist[0]))
for i in range(len(end_dist[0])-2,-1,-1):
    print(i)
    end_dp[:,i]=np.amax([end_dist[:,i],end_dp[:,i+1]],axis=0)

print(end_dp)
start_pos=np.argmax(start_dist*end_dp,axis=1) #(batch_size)
end_pos=map(lambda i:start_pos[i]+np.argmax(end_dist[i,start_pos[i]:]),range(len(end_dist)))


#--------------------------------------------------------------------------
"""
#tf.eye
A = tf.placeholder(tf.float32)
A_data = np.random.randn(2,6,6)

B = A - tf.eye(tf.shape(A)[1], batch_shape=[tf.shape(A)[0]])
#B = A - tf.eye(6,batch_shape=[2]) #this one works

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(B, feed_dict={A:A_data}))
    print(sess.run(B, feed_dict={A:A_data}).shape)
"""
#--------------------------------------------------------------------------
"""
#np.tensordow

A = np.random.randint(3, size=(2, 6, 5))
B = np.random.randint(3, size=(5, ))
print(np.tensordot(A, B, ((2), (0))))
"""
#--------------------------------------------------------------------------

"""
#tf.shape can handle None value
X1_data = np.random.randn(2,8)
X1 = tf.placeholder(tf.float32)
input_len = tf.shape(X1)[0]
input_len_2 = tf.reduce_sum(X1, axis=1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(input_len, feed_dict={X1:X1_data}))
    print(sess.run(input_len_2, feed_dict={X1:X1_data}))
"""
#--------------------------------------------------------------------------
"""
#tf.dynamic_rnn example
#Seems batch with different context_len tf.nn.dynamic_rnn can still handle

#output:
#(?, ?, 5)
#(?, ?, 5)
#['rnn/rnn/rnn_cell/kernel:0', 'rnn/rnn/rnn_cell/bias:0']


X1_data = np.random.randn(2,10,8)
X2_data = np.random.randn(2,15,8)

X1_lengths = [8, 4]
X2_lengths = [4, 14]

X1 = tf.placeholder(tf.float32, shape=[None, None, 8])
X2 = tf.placeholder(tf.float32, shape=[None, None, 5])
X1_mask = tf.placeholder(tf.float32, shape=[None, None])
X2_mask = tf.placeholder(tf.float32, shape=[None, None])

cell = tf.contrib.rnn.BasicRNNCell(num_units=5, name="rnn_cell")

for i in range(2):
    if i == 0:
        with tf.variable_scope("rnn"):
            outputs_1, last_states_1 = tf.nn.dynamic_rnn(cell=cell, inputs=X1 ,sequence_length=X1_lengths, dtype=tf.float32)
    else:
        with tf.variable_scope("rnn"):
            outputs_2, last_states_2 = tf.nn.dynamic_rnn(cell=cell, inputs=X2, sequence_length=X2_lengths, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    vars = [v.name for v in tf.trainable_variables()]
    print(sess.run([outputs_1, outputs_2], feed_dict={X1:X1_data, X2:X2_data}))
    print(outputs_1.shape)
    print(outputs_2.shape)

    #print(last_states_1)
    #print(last_states_2)
    print(vars)
"""




#--------------------------------------------------------------------------

#bilm-tf example
"""
#output:
context_input:  (2, 11, 1024)
question_input:  (1, 6, 1024)


from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# Location of pretrained LM.  Here we use the test fixtures.
datadir = "/Users/lam/Desktop/Lam-cs224n/Projects/qa/squad/data/elmo"
vocab_file = os.path.join(datadir, 'elmo_vocab.txt')
options_file = os.path.join(datadir, 'elmo.json')
weight_file = os.path.join(datadir, 'lm_weight.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
question_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)
question_embeddings_op = bilm(question_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_input = weight_layers(
        'input', question_embeddings_op, l2_coef=0.0
    )

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_output = weight_layers(
        'output', question_embeddings_op, l2_coef=0.0
    )


# Now we can compute embeddings.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_character_ids: context_ids,
                   question_character_ids: question_ids}
    )

    print("context_input: ", elmo_context_input_.shape)
    print("question_input: ",elmo_question_input_.shape)
"""