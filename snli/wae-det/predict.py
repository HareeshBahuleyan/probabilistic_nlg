from pathsetup import run_path_setup
run_path_setup()

import os
import gl
gl.isTrain = False

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

from det_wae import DetWAEModel
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

model = DetWAEModel(config,
                    embeddings_matrix,
                    word_index)
#----------------------------------------------------------------#

checkpoint = config['ckpt']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

#---------------------Reconstruction-----------------------------#
print("[INFO] Restoring model parameters ...")

preds = model.predict(checkpoint, x_test)
print('-'*100)

#----------------------------------------------------------------#

print("[INFO] Generate with test set input ...")
generated = ''
for pred in preds[:10]:
    generated += '\t\t' + ' '.join([model.idx_word[i] for i in pred if i not in [model.pad, model.eos]]) + '\n'
print(generated)

print('-'*100)
#----------------------------------------------------------------#

print("[INFO] Generate samples from the latent space ...")
model.random_sample(checkpoint)
model.random_sample_save(checkpoint, num_batches=3)

print('-'*100)
#----------------------------------------------------------------#

print("[INFO] Interpolate samples from the latent space ...")
model.linear_interpolate(checkpoint, num_samples=8)

print('-'*100)
#----------------------------------------------------------------#
