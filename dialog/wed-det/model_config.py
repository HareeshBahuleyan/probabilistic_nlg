import argparse
import gl
import os


def model_argparse():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--isDebug", type=bool, default=gl.isDebug, help='is debug')
    parser.add_argument("--device", type=str, default="0", help='tf device') # GPU 0 or 1
    parser.add_argument("--lstm_hidden_units", type=int, default=500, help='number of hidden units for the LSTM')
    parser.add_argument("--embedding_size", type=int, default=300, help='word embedding dimension')
    parser.add_argument("--num_layers", type=int, default=1, help='number of LSTM layers')

    parser.add_argument("--encoder_vocab", type=int, default=20000, help='encoder vocabulary size')
    parser.add_argument("--decoder_vocab", type=int, default=20000, help='decoder vocabulary size')
    parser.add_argument("--encoder_num_tokens", type=int, default=20, help='max number of words/tokens in the input sequence')
    parser.add_argument("--decoder_num_tokens", type=int, default=20, help='max number of words/tokens in the generated sequence')
    
    parser.add_argument("--latent_dim", type=int, default=300, help='dimension of z-latent space')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--n_epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--optimizer", type=str, default='adam', help='optimizer from: adam | sgd | rmsprop')

    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help='dropout keep probability')
    parser.add_argument("--initial_learning_rate", type=float, default=0.001, help='initial learning rate')
    parser.add_argument("--learning_rate_decay", type=float, default=0.98, help='learning rate decay')
    parser.add_argument("--min_learning_rate", type=float, default=0.00001, help='minimum learning rate')

    parser.add_argument("--lambda_val", type=float, default=3.0, help='initial value of lambda, i.e., MMD co-efficient')
    parser.add_argument("--kernel", type=str, default='IMQ', help='MMD loss based on kernel type from: IMQ or RBF ')
    
    parser.add_argument("--dataset", type=str, default='daily')
    parser.add_argument("--data_dir", type=str, default='../data/')
    parser.add_argument("--w2v_file", type=str, default='../w2v_models/w2vmodel_daily.pkl')
    parser.add_argument("--bleu_path", type=str, default='bleu/', help='path to save bleu scores')
    parser.add_argument("--model_checkpoint_dir", type=str, default='', help='path to save model checkpoints')
    parser.add_argument("--logs_dir", type=str, default='', help='path to save log files')

    parser.add_argument("--ckpt", type=str, default=None, help='checkpoint')
  
    args = parser.parse_args()
    config = vars(args)
    gl.config = config

    # Output log file
    gl.config_fingerprint = str(config['dataset']) + \
            '_lambdaWAE' + str(config['lambda_val']) + \
            '_batch' + str(config['batch_size']) + \
            '_kernel_' + str(config['kernel']) 

    if not gl.isTrain:
        return config

    # Create directories for saving model runs and stats
    pwd = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(pwd + '/bleu'):
        os.mkdir(pwd + '/bleu')
    
    if not os.path.exists(pwd + '/runs'):
        os.mkdir(pwd + '/runs')

    gl.log_writer = open(pwd + '/runs/log_' + gl.config_fingerprint, 'a')
    gl.log_writer.write(str(gl.config) + '\n')
    gl.log_writer.flush()

    # Model checkpoint
    if not os.path.exists(pwd + '/models'):
        os.mkdir(pwd + '/models')
    model_path = pwd + '/models/' + gl.config_fingerprint
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    config['model_checkpoint_dir'] = model_path + '/'

    # Model summary directory
    if not os.path.exists(pwd + '/summary_logs'):
        os.mkdir(pwd + '/summary_logs')
    summary_path = pwd + '/summary_logs/' + gl.config_fingerprint
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    config['logs_dir'] = summary_path

    return config
