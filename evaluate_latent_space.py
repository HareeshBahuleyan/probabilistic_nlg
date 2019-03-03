import argparse
import nltk
import random
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate latent space quality based on (1) AvgLen; (2) UnigramKL; (3) Entropy')
parser.add_argument('-ref', '--reference_path', help='Path to reference/corpus sentences (1 per line)', required=True)
parser.add_argument('-gen', '--generated_path', help='Path to reference/corpus sentences (1 per line)', required=True)
args = vars(parser.parse_args())


def get_avg_sent_lengths(sentences):
    """
    Computes the average number of tokens, given a list of sentences as input
    """
    sent_lens = [len(nltk.tokenize.word_tokenize(s)) for s in sentences]
    
    return np.mean(sent_lens)

def get_unigram_dist(sentences):
    """
    Returns a dictionary with unigrams and thier corresponding probability, given a list of sentences as input
    """
    # sentences = random.sample(sentences, sample_count)
    # sentences = [s[:-1] for s in sentences]
    tokens = nltk.tokenize.word_tokenize(' '.join(sentences))

    # Compute frequency distribution for all the unigrams in the text
    fdist = dict(nltk.FreqDist(tokens))
    total_unigram_count = len(tokens)

    for k in fdist.keys():
        fdist[k] = fdist[k]/total_unigram_count
        
    return fdist

def calc_discrete_entropy(fdist):
    """
    Given a unigram probability distribution (dictionary), this function computes the entropy = Sum P(i) * log(P(i))
    """
    entropy = 0    
    for token in fdist.keys():
        try:
            entropy += fdist[token] *  np.log(fdist[token])
        except KeyError: 
            print('Error token: ', token)
            pass
        
    return -entropy

def calc_discrete_kl(fdist_gen, fdist_true):
    """
    Given two unigram probability distributions, this function computes the KL divergence between them
    """
    kl_div = 0
    # KL(P|Q) = -1 *  Sum P(i) * log(Q(i)/P(i)) = Sum P(i) * log(P(i)/Q(i))
    for token in fdist_gen.keys():
        try:
            kl_div += fdist_gen[token] *  np.log(fdist_gen[token]/fdist_true[token])
        except KeyError: # If the word is not present in the training samples (i.e., true dist), i.e., Q(i)=0
            print('Error token: ', token)
            pass
        
    return kl_div

if __name__ == "__main__":
    
    with open (args['reference_path'], 'r') as f:
        ref_sentences = f.readlines()

    with open (args['generated_path'], 'r') as f:
        gen_sentences = f.readlines()    

    # Remove \n and unnecessary white spaces
    ref_sentences = [s.strip() for s in ref_sentences]
    gen_sentences = [s.strip() for s in gen_sentences]

    fdist_true = get_unigram_dist(ref_sentences)
    fdist_gen = get_unigram_dist(gen_sentences)

    print('-'*50)
    print('Entropy of Reference Sentences = {:.3f}'.format(calc_discrete_entropy(fdist_true)))
    print('Entropy of Generated Sentences = {:.3f}'.format(calc_discrete_entropy(fdist_gen)))
    print('-'*50)
    print('KL divergence KL(generated||reference) = {:.3f}'.format(calc_discrete_kl(fdist_gen, fdist_true)))
    print('-'*50)
    print('AverageLen of Reference Sentences = {:.2f}'.format(get_avg_sent_lengths(ref_sentences)))
    print('AverageLen of Generated Sentences = {:.2f}'.format(get_avg_sent_lengths(gen_sentences)))
    print('-'*50)
