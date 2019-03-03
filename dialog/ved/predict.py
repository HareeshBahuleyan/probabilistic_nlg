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
import pandas as pd
import utils

from ved import VEDModel
from sklearn.model_selection import train_test_split

np.random.seed(1337)

if config['dataset'] == 'daily':
    train_data = pd.read_csv(config['data_dir'] + 'DailyDial/de_duplicated/df_daily_train.csv')
    val_data = pd.read_csv(config['data_dir'] + 'DailyDial/de_duplicated/df_daily_valid_without_duplicates.csv')
    test_data = pd.read_csv(config['data_dir'] + 'DailyDial/de_duplicated/df_daily_test_without_duplicates.csv')
elif config['dataset'] == 'movie':
    train_data = pd.read_csv(config['data_dir'] + 'CornellMovieDialog/df_movie_train.csv')
    val_data = pd.read_csv(config['data_dir'] + 'CornellMovieDialog/df_movie_valid.csv')
    test_data = pd.read_csv(config['data_dir'] + 'CornellMovieDialog/df_movie_test.csv')
else:
    print('Invalid argument for --dataset !')
    exit()

input_sentences = pd.concat([train_data['line'], val_data['line'], test_data['line']])
output_sentences = pd.concat([train_data['reply'], val_data['reply'], test_data['reply']])

true_val = val_data['reply']
true_test = test_data['reply']
input_test = test_data['line']

filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
w2v_path = config['w2v_file']

print('[INFO] Tokenizing input and output sequences')
x, input_word_index = utils.tokenize_sequence(input_sentences, 
                                                filters, 
                                                config['encoder_num_tokens'], 
                                                config['encoder_vocab'])

y, output_word_index = utils.tokenize_sequence(output_sentences, 
                                                filters, 
                                                config['decoder_num_tokens'], 
                                                config['decoder_vocab'])

print('[INFO] Split data into train-validation-test sets')
dataset_sizes = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
x_train, y_train, x_val, y_val, x_test, y_test = utils.create_data_split(x, y, dataset_sizes)

encoder_embeddings_matrix = utils.create_embedding_matrix(input_word_index, 
                                                               config['embedding_size'], 
                                                               w2v_path)

decoder_embeddings_matrix = utils.create_embedding_matrix(output_word_index, 
                                                               config['embedding_size'], 
                                                               w2v_path)

# Re-calculate the vocab size based on the word_idx dictionary
config['encoder_vocab'] = len(input_word_index)
config['decoder_vocab'] = len(output_word_index)

#----------------------------------------------------------------#

model = VEDModel(config, 
                   encoder_embeddings_matrix, 
                   decoder_embeddings_matrix, 
                   input_word_index, 
                   output_word_index)

#----------------------------------------------------------------#

checkpoint = config['ckpt']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

#---------------------Reconstruction-----------------------------#
print("[INFO] Restoring model parameters ...")

preds = model.predict(checkpoint, 
                      x_test, 
                      y_test, 
                      true_test, 
                      )
print('-'*100)

#----------------------------------------------------------------#

print("[INFO] Generate with test set input ...")
generated = ''
for pred in preds[:10]:
    generated += '\t\t' + ' '.join([model.decoder_idx_word[i] for i in pred if i not in [model.pad, model.eos]]) + '\n'
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
