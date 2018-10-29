import os
import io
import json
import sys
import logging

import tensorflow as tf
from qa_model import QAModel
from tensorflow.python import debug as tfbdg
#from official_eval_helper import get_json_data, generate_answers
from vocab import get_glove, get_char_embed


logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .. (/Users/lam/Desktop/Lam-cs224n/Projects/squad2)
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") #relative path of data_dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") #relative path of experiments_dir

#High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have mutiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for experiment. Will create experiments/ directory to store data for a experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs for training. 0 means infinite")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning_rate")
tf.app.flags.DEFINE_float("max_gradient_norm", 7.0, "Clip gradient to this norm")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch_size to use") #default is 100
tf.app.flags.DEFINE_integer("hidden_size", 75, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 350, "The maximum context length of the model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of the model")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")
#for chraceter cnn
tf.app.flags.DEFINE_integer("char_size", 20, "Size of the char embedding")
tf.app.flags.DEFINE_integer("num_of_char", 72, "Total number of characters")
tf.app.flags.DEFINE_integer("max_word_len", 20, "Max number of characters of a word")

#Save, print, eval every
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do printing")
tf.app.flags.DEFINE_integer("save_every", 1000, "How many iterations to do saving")
tf.app.flags.DEFINE_integer("eval_every", 1000, "How many iterations to do evaluation")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

#Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.840B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("elmo_dir", os.path.join(DEFAULT_DATA_DIR, "elmo"), "Where to find preprocessed ELMo data for training. Defaults to data/elmo")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

#Optimization
tf.app.flags.DEFINE_boolean("USE_CUDNN", False, "Decide if use cudnn lstm")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu) #set how many gpu to use in environment variables


def initialize_model(session, model, train_dir, expect_exists):
    #First check if there exist trained model in the path (ckpt.model_check_path / v2_path)
    #If yes then restore the model
    #If no then initialise the fresh parameters

    print("Looking for model at %s..." %train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir) #return a class or None
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" %ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint in %s" %train_dir)
        else:
            print ("There is no saved checkpoint in %s. Creating model with fresh parameters" %train_dir)
            session.run(tf.global_variables_initializer())
            print("Num params: %d" % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    model.feed_embedding(session)
def main(unused_argv):
    #First check the FLAGS enter correctly (format), python version and tensorflow version
    #Then check if train_dir or experiment_dir defined
    #set bestmodel path which named best_checkpoint
    #set context, question. ans path
    #read glove
    #Initialise the model architecture
    #gpu setting
    #mode choice

    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)
    if sys.version_info[0] != 3:
        raise Exception("ERROR: You must use Python 3 but you are running Python %i" % sys.version_info[0])

    print ("This code was developed and tested on TensorFlow 1.8.0. Your TensorFlow version: %s" % tf.__version__)
    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
    
    # Define train_dir
    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    #glove path
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)
    char2id, id2char = get_char_embed()

    #path for context, question, ans_span
    train_context_path = os.path.join(FLAGS.data_dir, "train.context")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
    dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

    #Initialise the model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix, id2char, char2id)

    # Some GPU settings
    config = tf.ConfigProto() #set configuration for sess.run
    config.gpu_options.allow_growth = True #make gpu storage usage based for condition

    #different modes
    if FLAGS.mode == "train":
        #setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.flag_values_dict(), fout) #NoteL changed from FLAGS.__flags to FLAGS.flag_values_dict() after tensorflow 1.5

        # Make bestmodel dir
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:
            #Added tfdbg
            

            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            #Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)


    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)

            """
    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print ("Writing predictions to %s..." % FLAGS.json_out_path)
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print ("Wrote predictions to %s" % FLAGS.json_out_path)

    """
    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()






