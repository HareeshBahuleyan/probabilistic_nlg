from pathsetup import run_path_setup
run_path_setup()

import time
import pickle
import tensorflow as tf
import numpy as np
import utils
import gl
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from tensorflow.python.layers.core import Dense
from decoder import basic_decoder


class StochasticWAEModel(object):

    def __init__(self, config, embeddings_matrix, word_index):

        self.config = config

        self.lstm_hidden_units = config['lstm_hidden_units']
        self.embedding_size = config['embedding_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        
        self.lambda_val = config['lambda_val']
        self.gamma_kl = config['gammaKL']

        self.vocab_size = config['vocab_size']
        self.num_tokens = config['num_tokens']

        self.dropout_keep_prob = config['dropout_keep_prob']
        self.z_temp = config['z_temp']

        self.initial_learning_rate = config['initial_learning_rate']
        self.learning_rate_decay = config['learning_rate_decay']
        self.min_learning_rate = config['min_learning_rate']

        self.batch_size = config['batch_size']
        self.epochs = config['n_epochs']

        self.embeddings_matrix = embeddings_matrix
        self.word_index = word_index
        self.idx_word = dict((i, word) for word, i in word_index.items())

        self.logs_dir = config['logs_dir']
        self.model_checkpoint_dir = config['model_checkpoint_dir']
        self.bleu_path = config['bleu_path']

        self.pad = self.word_index['PAD']
        self.eos = self.word_index['EOS']
        self.unk = self.word_index['UNK']
        
        self.epoch_bleu_score_val = {'1': [], '2': [], '3': [], '4': []}
        self.log_str = []

        self.build_model()

    def build_model(self):
        print("[INFO] Building Model ...")

        self.init_placeholders()
        self.embedding_layer()
        self.build_encoder()
        self.build_latent_space()
        self.sample_gaussian()
        self.build_decoder()
        self.loss()
        self.optimize()
        self.summary()

    def init_placeholders(self):
        with tf.name_scope("model_inputs"):
            # Create palceholders for inputs to the model
            self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_tokens], name='input') # batch x maxlen
            self.target_data = tf.placeholder(tf.int32, [self.batch_size, self.num_tokens], name='targets') # batch x maxlen
            self.lr = tf.placeholder(tf.float32, name='learning_rate', shape=())
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # Dropout Keep Probability
            self.source_sentence_length = tf.placeholder(tf.int32, shape=(self.batch_size,),
                                                         name='source_sentence_length') # batch
            self.target_sentence_length = tf.placeholder(tf.int32, shape=(self.batch_size,),
                                                         name='target_sentence_length') # batch
            self.lambda_coeff = tf.placeholder(tf.float32, name='lambda_coeff', shape=())
            self.z_temperature = tf.placeholder(tf.float32, name='z_temperature', shape=())

    def embedding_layer(self):
        with tf.name_scope("word_embeddings"):
            self.embeddings = tf.Variable(
                initial_value=np.array(self.embeddings_matrix, dtype=np.float32),
                dtype=tf.float32, trainable=False)
            self.enc_embed_input = tf.nn.embedding_lookup(self.embeddings, self.input_data) # batch x maxlen x embed_dim
            self.enc_embed_input = self.enc_embed_input[:, :tf.reduce_max(self.source_sentence_length), :]

            with tf.name_scope("decoder_inputs"):
                # shifted = tf.strided_slice(self.target_data, [0, 0], [self.batch_size, -1], [1, 1],
                #                          name='slice_input')  # Minus 1 implies everything till the last dim
                shifted = self.target_data[:,:-1] # batch x (maxlen - 1)
                self.dec_input = tf.concat([tf.fill([self.batch_size, 1], self.word_index['GO']), shifted], 1,
                                           name='dec_input') # batch x maxlen
                self.dec_embed_input = tf.nn.embedding_lookup(self.embeddings, self.dec_input)
                self.max_tar_len = tf.reduce_max(self.target_sentence_length)
                self.dec_embed_input = self.dec_embed_input[:, :self.max_tar_len, :] # batch x maxlen x embed_dim
                # self.dec_embed_input = tf.nn.dropout(self.dec_embed_input, keep_prob=self.keep_prob)

    def build_encoder(self):
        with tf.name_scope("encode"):
            for layer in range(self.num_layers):
                with tf.variable_scope('encoder_{}'.format(layer + 1)):
                    cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_units)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob)

                    cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_units)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob)

                    self.enc_output, self.enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                      cell_bw,
                                                                                      self.enc_embed_input,
                                                                                      self.source_sentence_length,
                                                                                      dtype=tf.float32)

            # Join outputs since we are using a bidirectional RNN
            self.h_N = tf.concat([self.enc_state[0][1], self.enc_state[1][1]], axis=-1,
                                 name='h_N')  # Concatenated h from the fw and bw LSTMs
            self.enc_outputs = tf.concat([self.enc_output[0], self.enc_output[1]], axis=-1, name='encoder_outputs')
            
    def build_latent_space(self):
        with tf.name_scope("latent_space"):
            self.z_mean = Dense(self.latent_dim, name='z_mean')(self.h_N)
            self.z_log_sigma = Dense(self.latent_dim, name='z_log_sigma')(self.h_N)

            self.z_tilda = tf.identity(self.sample_z_tilda_from_posterior(), name='z_tilda')

    def sample_z_tilda_from_posterior(self):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope('sample_gaussian'):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(self.z_log_sigma), name='epsilon')
            return self.z_mean + tf.scalar_mul(self.z_temperature,
                                               epsilon * tf.exp(self.z_log_sigma))  # N(mu, I * sigma**2)

    def calculate_kl_loss(self):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # KL divergence between N(mu, sigma) vs. N(mu, I)
            p = tf.contrib.distributions.Normal(loc=self.z_mean, scale=tf.exp(self.z_log_sigma)) # Posterior
            q = tf.contrib.distributions.Normal(loc=self.z_mean, scale=tf.ones(shape = self.z_log_sigma.shape)) # Prior
            kl_div = tf.reduce_sum(tf.contrib.distributions.kl_divergence(p, q), axis=-1)
            
            return kl_div

    def sample_gaussian(self):
        with tf.name_scope('sample_gaussian'):
            # Random sample from Gaussian prior
            self.z_sampled = tf.random_normal([self.batch_size, self.latent_dim], name='z_sampled') # Dimension [batch_size x latent_dim]

    def build_decoder(self):
        with tf.variable_scope("decode"):
            for layer in range(self.num_layers):
                with tf.variable_scope('decoder_{}'.format(layer + 1)):
                    dec_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2 * self.lstm_hidden_units)
                    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=self.keep_prob)

            self.output_layer = Dense(self.vocab_size)

            self.init_state = dec_cell.zero_state(self.batch_size, tf.float32)

            with tf.name_scope("training_decoder"):
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.dec_embed_input,
                                                                    sequence_length=self.target_sentence_length,
                                                                    time_major=False)

                training_decoder = basic_decoder.BasicDecoder(dec_cell,
                                                              training_helper,
                                                              initial_state=self.init_state,
                                                              latent_vector=self.z_tilda,
                                                              output_layer=self.output_layer)

                self.training_logits, _state, _len = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                       output_time_major=False,
                                                                                       impute_finished=True,
                                                                                       maximum_iterations=self.num_tokens)

                self.training_logits = tf.identity(self.training_logits.rnn_output, 'logits')

            with tf.name_scope("validate_decoder"):
                start_token = self.word_index['GO']
                end_token = self.word_index['EOS']

                start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size],
                                       name='start_tokens')

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                                            start_tokens,
                                                                            end_token)

                inference_decoder = basic_decoder.BasicDecoder(dec_cell,
                                                               inference_helper,
                                                               initial_state=self.init_state,
                                                               latent_vector=self.z_tilda,
                                                               output_layer=self.output_layer)

                self.validate_logits, _state, _len = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                        output_time_major=False,
                                                                                        impute_finished=True,
                                                                                        maximum_iterations=self.num_tokens)


                self.validate_sent = tf.identity(self.validate_logits.sample_id, name='predictions')

            with tf.name_scope("inference_decoder"):
                start_token = self.word_index['GO']
                end_token = self.word_index['EOS']

                start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size],
                                       name='start_tokens')

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                                            start_tokens,
                                                                            end_token)

                inference_decoder = basic_decoder.BasicDecoder(dec_cell,
                                                               inference_helper,
                                                               initial_state=self.init_state,
                                                               latent_vector=self.z_sampled,
                                                               output_layer=self.output_layer)

                self.inference_logits, _state, _len = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                        output_time_major=False,
                                                                                        impute_finished=True,
                                                                                        maximum_iterations=self.num_tokens)

                self.inference_logits = tf.identity(self.inference_logits.sample_id, name='predictions')

    def mmd_penalty(self, sample_qz, sample_pz):
        n = self.batch_size
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if self.config['kernel'] == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # if opts['verbose']:
            #     sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif self.config['kernel'] == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            #if opts['pz'] == 'normal':
            #    Cbase = 2. * opts['zdim'] * sigma2_p
            #elif opts['pz'] == 'sphere':
            #    Cbase = 2.
            #elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
            #    Cbase = opts['zdim']

            Cbase = 2. * self.config['latent_dim'] * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat

    def loss(self):
        with tf.name_scope('losses'):
            self.wasserstein_loss = self.mmd_penalty(self.z_sampled, self.z_tilda)
            self.kl_regularization_loss = self.calculate_kl_loss() # KL loss on the stochastically encoded z, so that it is not peaked

            # Create the weights for sequence_loss
            masks = tf.sequence_mask(self.target_sentence_length, self.num_tokens, dtype=tf.float32, name='masks')

            self.xent_loss = tf.contrib.seq2seq.sequence_loss(
                self.training_logits,
                self.target_data[:, :self.max_tar_len],
                weights=masks[:, :self.max_tar_len],
                average_across_batch=True)

            # L2-Regularization
            self.var_list = tf.trainable_variables()
            self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list if 'bias' not in v.name]) * 0.001

            self.cost = self.xent_loss + self.config['lambda_val'] * self.wasserstein_loss + self.gamma_kl * self.kl_regularization_loss # + self.lossL2

    def optimize(self):
        # Optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            # optimizer = tf.train.GradientDescentOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost, var_list=self.var_list)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('xent_loss', tf.reduce_sum(self.xent_loss))
            tf.summary.scalar('l2_loss', tf.reduce_sum(self.lossL2))
            tf.summary.scalar("wasserstein_loss", tf.reduce_sum(self.wasserstein_loss))
            tf.summary.scalar('total_loss', tf.reduce_sum(self.cost))
            tf.summary.scalar('kl_regularization_loss', tf.reduce_sum(self.kl_regularization_loss))
            tf.summary.scalar('lambda', self.lambda_coeff)

            self.summary_op = tf.summary.merge_all()

    def monitor(self, x_val, sess, epoch_i, time_consumption):
        self.validate(sess, x_val)
        val_bleu_str = str(self.epoch_bleu_score_val['1'][-1]) + ' | ' \
                       + str(self.epoch_bleu_score_val['2'][-1]) + ' | ' \
                       + str(self.epoch_bleu_score_val['3'][-1]) + ' | ' \
                       + str(self.epoch_bleu_score_val['4'][-1])

        val_str = '\t\t Generated \t|\t Actual \n'
        for pred, ref in zip(self.val_pred[:20], self.val_ref[:20]):
            val_str += '\t\t' + pred + '\t|\t' + ref + '\n'

        print(val_str)
        gl.log_writer.write(val_str)

        generated = self.random_sample_in_session(sess)

        print(generated)
        gl.log_writer.write(generated)

        log_thisepoch = 'Epoch {:>3}/{} - Time {:>6.1f}, Train loss: {:>3.2f}, Val BLEU: {}\n\n'.format(epoch_i,
                                                                                                        self.epochs,
                                                                                                        time_consumption,
                                                                                                        self.train_xent,
                                                                                                        val_bleu_str)

        print(log_thisepoch)
        gl.log_writer.write(log_thisepoch)
        gl.log_writer.flush()

        saver = tf.train.Saver()
        saver.save(sess, self.model_checkpoint_dir + str(epoch_i) + ".ckpt")

        # Save the validation BLEU scores so far
        with open(self.bleu_path + gl.config_fingerprint + '.pkl', 'wb') as f:
            pickle.dump(self.epoch_bleu_score_val, f)

        self.log_str.append(log_thisepoch)

        with open('bleu_logs.txt', 'w') as f:
            f.write('\n'.join(self.log_str))

    def train(self, x_train, x_val):

        print('[INFO] Training process started')

        learning_rate = self.initial_learning_rate
        iter_i = 0

        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

            for epoch_i in range(1, self.epochs + 1):

                start_time = time.time()
                for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                        utils.get_batches(x_train, self.batch_size)):

                    try:
                        iter_i += 1

                        _, _summary, self.train_xent = sess.run(
                            [self.train_op, self.summary_op, self.xent_loss],
                            feed_dict={self.input_data: input_batch, # <batch x maxlen>
                                       self.target_data: output_batch, # <batch x maxlen>
                                       self.lr: learning_rate,
                                       self.source_sentence_length: sent_lengths,
                                       self.target_sentence_length: sent_lengths,
                                       self.keep_prob: self.dropout_keep_prob,
                                       self.z_temperature: self.z_temp,
                                       self.lambda_coeff: self.lambda_val,
                                       })

                        writer.add_summary(_summary, iter_i)

                    except Exception as e:
                        print(iter_i, e)
                        exit(1)
                        pass

                # Reduce learning rate, but not below its minimum value
                learning_rate = np.max([self.min_learning_rate, learning_rate * self.learning_rate_decay])
                time_consumption = time.time() - start_time
                self.monitor(x_val, sess, epoch_i, time_consumption)

    def validate(self, sess, x_val):
        # Calculate BLEU on validation data
        hypotheses_val = []
        references_val = []

        for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                utils.get_batches(x_val, self.batch_size)):
            pred_sentences, self._validate_logits = sess.run(
                                     [self.validate_sent, self.validate_logits],
                                     feed_dict={self.input_data: input_batch,
                                                self.source_sentence_length: sent_lengths,
                                                self.keep_prob: 1.0,
                                                self.z_temperature: self.z_temp,
                                                })


            for pred, actual in zip(pred_sentences, output_batch):
                hypotheses_val.append(
                    word_tokenize(
                        " ".join([self.idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])))
                references_val.append(
                    [word_tokenize(" ".join([self.idx_word[i] for i in actual if i not in [self.pad, -1, self.eos]]))])
            self.val_pred = ([" ".join(sent)    for sent in hypotheses_val])
            self.val_ref  = ([" ".join(sent[0]) for sent in references_val])

        bleu_scores = utils.calculate_bleu_scores(references_val, hypotheses_val)

        self.epoch_bleu_score_val['1'].append(bleu_scores[0])
        self.epoch_bleu_score_val['2'].append(bleu_scores[1])
        self.epoch_bleu_score_val['3'].append(bleu_scores[2])
        self.epoch_bleu_score_val['4'].append(bleu_scores[3])

    def predict(self, checkpoint, x_test):
        pred_logits = []
        hypotheses_test = []
        references_test = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                    utils.get_batches(x_test, self.batch_size)):
                result = sess.run(self.validate_sent, feed_dict={self.input_data: input_batch,
                                                                    self.source_sentence_length: sent_lengths,
                                                                    self.keep_prob: 1.0,
                                                                    self.z_temperature: self.z_temp,
                                                                    })

                pred_logits.extend(result)

                for pred, actual in zip(result, output_batch):
                    hypotheses_test.append(
                        word_tokenize(" ".join(
                            [self.idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])))
                    references_test.append([word_tokenize(
                        " ".join([self.idx_word[i] for i in actual if i not in [self.pad, -1, self.eos]]))])

            bleu_scores = utils.calculate_bleu_scores(references_test, hypotheses_test)

        print('BLEU 1 to 4 : {}'.format(' | '.join(map(str, bleu_scores))))

        return pred_logits

    def show_output_sentences(self, preds, x_test):
        for pred, actual in zip(preds, x_test):
            # Actual and generated
            print('A: {}'.format(
                " ".join([self.idx_word[i] for i in actual if i not in [self.pad, self.eos]])))
            print('G: {}\n'.format(
                " ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]])))

    def get_diversity_metrics(self, checkpoint, x_test, num_samples=10, num_iterations=3):

        x_test_repeated = np.repeat(x_test, num_samples, axis=0)

        entropy_list  = []
        uni_diversity = []
        bi_diversity  = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for _ in tqdm(range(num_iterations)):
                total_ent = 0
                uni = 0
                bi = 0
                answer_logits = []
                pred_sentences = []

                for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                        utils.get_batches(x_test_repeated, self.batch_size)):
                    result = sess.run(self.validate_sent, feed_dict={self.input_data: input_batch,
                                                                        self.source_sentence_length: sent_lengths,
                                                                        self.keep_prob: 1.0,
                                                                        self.z_temperature: self.z_temp,
                                                                        })
                    answer_logits.extend(result)

                for idx, (actual, pred) in enumerate(zip(x_test_repeated, answer_logits)):
                    pred_sentences.append(" ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]]))

                    if (idx + 1) % num_samples == 0:
                        word_list = [word_tokenize(p) for p in pred_sentences]
                        corpus = [item for sublist in word_list for item in sublist]
                        total_ent += utils.calculate_entropy(corpus)
                        diversity_result = utils.calculate_ngram_diversity(corpus)
                        uni += diversity_result[0]
                        bi += diversity_result[1]

                        pred_sentences = []

                entropy_list.append(total_ent / len(x_test))
                uni_diversity.append(uni / len(x_test))
                bi_diversity.append(bi / len(x_test))

        print('Entropy = {:>.3f} | Distinct-1 = {:>.3f} | Distinct-2 = {:>.3f}'.format(np.mean(entropy_list),
                                                                                       np.mean(uni_diversity),
                                                                                       np.mean(bi_diversity)))
    
    def random_sample(self, checkpoint):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            
            z_sampled = np.random.normal(size=(self.batch_size, self.latent_dim))
            result = sess.run(self.inference_logits,
                                feed_dict={self.z_sampled: z_sampled,
                                            self.keep_prob: 1.0,
                                            })

            for pred in result:
                sent = " ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]])
                print('G: {}'.format(sent))

    def random_sample_in_session(self, sess):
        z_sampled = np.random.normal(size=(self.batch_size, self.latent_dim))
        result = sess.run(self.inference_logits,feed_dict={self.z_sampled: z_sampled,self.keep_prob: 1.0,})

        generated = ''

        for pred in result[:10]:
            generated += '\t\t' + ' '.join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]]) + '\n'
        return generated

    def random_sample_save(self, checkpoint, num_batches=1):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            gen_samples = []

            for i in range(num_batches):
                z_sampled = np.random.normal(size=(self.batch_size, self.latent_dim))
                result = sess.run(self.inference_logits,
                                  feed_dict={self.z_sampled: z_sampled,
                                             self.keep_prob: 1.0,
                                             })

                for pred in result:
                    sent = " ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]])
                    gen_samples.append(sent)

        # Create directories for saving sentences generated by random sampling
        pwd = os.path.dirname(os.path.realpath(__file__))

        if not os.path.exists(pwd + '/samples'):
            os.mkdir(pwd + '/samples')
        
        with open(pwd + '/samples/' + 'sample.txt', 'w') as f:
            f.write('\n'.join(gen_samples))
                
    def linear_interpolate(self, checkpoint, num_samples):
        sampled = []
        for i in range(self.batch_size // num_samples):
            z = np.random.normal(0, 1, (2, self.latent_dim))
            s1_z = z[0]
            s2_z = z[1]
            s1_z = np.repeat(s1_z[None, :], num_samples, axis=0)
            s2_z = np.repeat(s2_z[None, :], num_samples, axis=0)
            steps = np.linspace(0, 1, num_samples)[:, None]
            sampled.append(s1_z * (1 - steps) + s2_z * steps)

        sampled = np.reshape(np.array(sampled), newshape=(self.batch_size, self.latent_dim))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            result = sess.run(self.inference_logits,
                              feed_dict={self.z_sampled: sampled,
                                         self.keep_prob: 1.0,
                                         })

            for i, pred in enumerate(result):
                if i % num_samples == 0:
                    print()
                print('G: {}'.format(
                    " ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]])))
                
    def linear_interpolate_between_inputs(self, checkpoint, start_sent, end_sent, num_samples=8):

        # Convert seq of words to seq of indices
        # if the word is not present, use default in get(): UNK
        start_sent = word_tokenize(start_sent)
        end_sent = word_tokenize(end_sent)
        start_idx_seq = [self.word_index.get(word, self.unk) for word in start_sent] + [self.eos] 
        end_idx_seq = [self.word_index.get(word, self.unk) for word in end_sent] + [self.eos]  # Append EOS token
        start_idx_seq = np.concatenate([start_idx_seq, np.zeros(max(0, self.num_tokens - len(start_idx_seq)))])[
                      :self.num_tokens]
        end_idx_seq = np.concatenate([end_idx_seq, np.zeros(max(0, self.num_tokens - len(end_idx_seq)))])[
                        :self.num_tokens]

        # Reshape/tile so that the input has first dimension as batch size
        inp_idx_seq = np.tile(np.vstack([start_idx_seq, end_idx_seq]), [self.batch_size//2, 1])
        # source_sent_lengths = [np.count_nonzero(seq) for seq in inp_idx_seq]

        # Get z_vector of first and last sentence
        z_vecs = self.get_zvector(checkpoint, inp_idx_seq)

        sampled = []
        s1_z = z_vecs[0]
        s2_z = z_vecs[1]
        s1_z = np.repeat(s1_z[None, :], num_samples, axis=0)
        s2_z = np.repeat(s2_z[None, :], num_samples, axis=0)
        steps = np.linspace(0, 1, num_samples)[:, None]
        sampled.append(s1_z * (1 - steps) + s2_z * steps)

        sampled = np.tile(sampled[0], [self.batch_size//num_samples, 1])
        # sampled = np.reshape(np.array(sampled), newshape=(self.batch_size, self.latent_dim))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            result = sess.run(self.inference_logits,
                              feed_dict={self.z_sampled: sampled,
                                         self.keep_prob: 1.0,
                                         })

            for i, pred in enumerate(result[:num_samples]):
                print('G: {}'.format(
                    " ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]])))

    def get_neighbourhood(self, checkpoint, x_test, temp=1.0, num_samples=10):
        answer_logits = []
        pred_sentences = []
        x_test_repeated = np.repeat(x_test, num_samples, axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                    utils.get_batches(x_test_repeated, self.batch_size)):
                result = sess.run(self.validate_sent, feed_dict={self.input_data: input_batch,
                                                                    self.source_sentence_length: sent_lengths,
                                                                    self.keep_prob: 1.0,
                                                                    self.z_temperature: temp,
                                                                    })
                answer_logits.extend(result)

            for idx, (actual, pred) in enumerate(zip(x_test_repeated, answer_logits)):
                pred_sentences.append(" ".join([self.idx_word[i] for i in pred if i not in [self.pad, self.eos]]))

        for j in range(len(pred_sentences)):
            if j % num_samples == 0:
                print('\nA: {}'.format(" ".join([self.idx_word[i] for i in x_test_repeated[j] if i not in [self.pad, self.eos]])))
            print('G: {}'.format(pred_sentences[j]))

    def get_zvector(self, checkpoint, x_test):
        z_vecs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                    utils.get_batches(x_test, self.batch_size)):
                result = sess.run(self.z_mean, feed_dict={self.input_data: input_batch,
                                                           self.source_sentence_length: sent_lengths,
                                                           self.keep_prob: 1.0,
                                                           })
                z_vecs.extend(result)

        return np.array(z_vecs)
    
    def get_z_log_sigma(self, checkpoint, x_test):
        z_vecs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for batch_i, (input_batch, output_batch, sent_lengths) in enumerate(
                    utils.get_batches(x_test, self.batch_size)):
                result = sess.run(self.z_log_sigma, feed_dict={self.input_data: input_batch,
                                                           self.source_sentence_length: sent_lengths,
                                                           self.keep_prob: 1.0,
                                                           })
                z_vecs.extend(result)

        return np.array(z_vecs)

    