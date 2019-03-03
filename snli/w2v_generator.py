from pathsetup import run_path_setup
run_path_setup()

import os
import gensim
import numpy as np
from nltk.tokenize import word_tokenize
import utils

def main():

    snli_data = utils.get_sentences(file_path='data/snli_sentences_all.txt')

    print('[INFO] Number of sentences = {}'.format(len(snli_data)))

    sentences = [s.strip() for s in snli_data]

    np.random.shuffle(sentences)
    sentences = [word_tokenize(s) for s in sentences]
    w2v_model = gensim.models.Word2Vec(sentences,
                                       size=300,
                                       min_count=1,
                                       iter=50)
    if not os.path.exists('w2v_models'):
        os.mkdir('w2v_models')

    w2v_model.save('w2v_models/w2v_300d_snli_all_sentences.pkl')
    print('[INFO] Word embeddings pre-trained successfully')


if __name__ == '__main__':
    main()