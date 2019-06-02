import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias, _p

from lstm_layer import param_init_lstm, lstm_layer

""" init. parameters. """  
def init_params(options,W):
    
    n_words = options['n_words']
    n_x = options['n_x']  
    n_h = options['n_h']
    n_z = options['n_z']
    n_s = options['n_s']
    
    params = OrderedDict()
    # word embedding init or load
    # params['Wemb'] = uniform_weight(n_words,n_x)
    params['Wemb'] = W.astype(config.floatX)
    
    options[_p('lstm', 'n_x')] = n_x
    options[_p('lstm', 'n_h')] = n_h
    options[_p('lstm', 'n_g')] = n_s
    
    params = param_init_lstm(options,params, 'lstm')
    
    params['Vhid'] = uniform_weight(n_h,n_x)
    params['bhid'] = zero_bias(n_words) 
    
    params['bos'] = zero_bias(n_x)
    params['Tv'] = uniform_weight(n_z, n_h)
    params['Ts'] = uniform_weight(n_x, n_h)
    params['Ta'] = uniform_weight(n_s, n_h)
    
    #re-z
    params['task_W'] = uniform_weight(n_h, n_z)
    params['task_b'] = zero_bias(n_z)

    return params

def to_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
""" Building model... """

def build_model(tparams,options):
    
    trng = RandomStreams(options['SEED'])
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(options['dropout_val']))

    # input sentences, size of n_steps * n_samples
    x = tensor.matrix('x', dtype='int64')
    # the corresponding masks padding zeros
    mask = tensor.matrix('mask', dtype=config.floatX)
    # size of n_samples * n_z
    z = tensor.tensor3('z', dtype=config.floatX)
    z_score = tensor.matrix('z_score', dtype=config.floatX)
    s = tensor.matrix('s', dtype=config.floatX)
    z1 = dropout(z, trng, use_noise)
    s = dropout(s, trng, use_noise)

    n_steps = x.shape[0] # the sentence length in this mini-batch
    n_samples = x.shape[1] # the number of sentences in this mini-batch
    n_x = tparams['Wemb'].shape[1] # the dimension of the word embedding
    
    # size of n_steps,n_samples,n_x
    emb = tparams['Wemb'][x.flatten()].reshape([n_steps,n_samples,n_x])
    emb = dropout(emb, trng, use_noise)
    
    # visual feature: 1 * n_samples * n_x
    z0 = tensor.dot(z1[:,0,:],tparams['Tv']).dimshuffle('x',0,1)
    bos = (tensor.dot(tparams['bos'],tparams['Ts']) + tensor.dot(s,tparams['Ta'])).dimshuffle('x',0,1)
    # n_steps * n_samples * n_x
	# feed visual feature z only at the first step.
    emb_input = tensor.dot(emb,tparams['Ts']) + tensor.dot(s,tparams['Ta'])
    emb_input = tensor.concatenate((z0, bos, emb_input[:n_steps-1]))

    # n_steps * n_samples
	# z has the same mask as the first word.
    mask0 =mask[0].dimshuffle('x',0)
    mask1 =mask[0].dimshuffle('x',0)
    mask_input = tensor.concatenate((mask0, mask1, mask[:n_steps-1]))

    # decoding the sentence vector z back into the original sentence
    h_decoder, _ = lstm_layer(tparams, emb_input, mask_input, prefix='lstm')
    h_decoder = dropout(h_decoder, trng, use_noise)
    
    h_decoder = h_decoder[1:]
    
    #re-z
    
    pred_z = tensor.tanh(tensor.dot(h_decoder, tparams['task_W'])+ tparams['task_b'])
    pred_z_cos = z_score[None,:,:] - (pred_z[:,:,None,:] * z[None,:,:,:]).sum(axis = 3) / tensor.sqrt(tensor.sqr(pred_z).sum(axis = 2))[:,:,None] / tensor.sqrt(tensor.sqr(z).sum(axis = 2))[None,:,:]
    
    shape = h_decoder.shape
    h_decoder = h_decoder.reshape((shape[0]*shape[1], shape[2]))
    
    Vhid = tensor.dot(tparams['Vhid'],tparams['Wemb'].T)
    pred_x = tensor.dot(h_decoder, Vhid) + tparams['bhid']
    pred = tensor.nnet.softmax(pred_x)
    x_vec = x.reshape((shape[0]*shape[1],))
    
    index = tensor.arange(shape[0]*shape[1])
    
    pred_word = pred[index, x_vec]
    mask_word = mask.reshape((shape[0]*shape[1],))
    
    index_list = theano.tensor.eq(mask_word, 1.).nonzero()[0]
    
    pred_word = pred_word[index_list]
    
    # the cross-entropy loss
    cost1 = -tensor.log(pred_word + 1e-6).sum() / n_samples

    # the cosine hinge loss
    pred_z_cos = pred_z_cos.reshape((shape[0]*shape[1], pred_z_cos.shape[2]))
    pred_z_cos = tensor.maximum(pred_z_cos[index_list], 0.)
    cost2 = pred_z_cos.sum() / n_samples
    
    cost = cost1 + cost2

    return use_noise, x, mask, s, z, z_score, cost, cost1, cost2
