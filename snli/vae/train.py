from pathsetup import run_path_setup
run_path_setup()

import os
import gl
gl.isTrain = True

from model_config import model_argparse
config = model_argparse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import numpy as np
import utils

from vae import VAEModel
from sklearn.model_selection import train_test_split

np.random.seed(1337)

snli_data = utils.get_sentences(file_path = config['data'])

print('[INFO] Number of sentences = {}'.format(len(snli_data)))

sentences = [s.strip() for s in snli_data]

np.random.shuffle(sentences)

print('[INFO] Tokenizing input and output sequences')
filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
x, word_index = utils.tokenize_sequence(sentences,
                                             filters,
                                             config['num_tokens'],
                                             config['vocab_size'])

print('[INFO] Split data into train-validation-test sets')
x_train, _x_val_test = train_test_split(x, test_size = 0.1, random_state = 10)
x_val, x_test = train_test_split(_x_val_test, test_size = 0.5, random_state = 10)

w2v = config['w2v_file']
embeddings_matrix = utils.create_embedding_matrix(word_index,
                                                  config['embedding_size'],
                                                  w2v)

# Re-calculate the vocab size based on the word_idx dictionary
config['vocab_size'] = len(word_index)

#----------------------------------------------------------------#

model = VAEModel(config, 
                    embeddings_matrix,
                    word_index)

model.train(x_train, x_val)

gl.log_writer.close()

#----------------------------------------------------------------#

