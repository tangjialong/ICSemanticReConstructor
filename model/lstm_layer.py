import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import uniform_weight, zero_bias, ortho_weight

def param_init_lstm(options, params, prefix):
    
    n_x = options[_p(prefix, 'n_x')]
    n_h = options[_p(prefix, 'n_h')]
    n_g = options[_p(prefix, 'n_g')]

    params[_p(prefix, 'W_i')] = ortho_weight(n_h)
    params[_p(prefix, 'W_f')] = ortho_weight(n_h)
    params[_p(prefix, 'W_o')] = ortho_weight(n_h)
    params[_p(prefix, 'W_c')] = ortho_weight(n_h)
    
    params[_p(prefix, 'U_i')] = ortho_weight(n_h)
    params[_p(prefix, 'U_f')] = ortho_weight(n_h)
    params[_p(prefix, 'U_o')] = ortho_weight(n_h)
    params[_p(prefix, 'U_c')] = ortho_weight(n_h)
    
    params[_p(prefix,'b_i')] = zero_bias(n_h)
    params[_p(prefix,'b_f')] = zero_bias(n_h)
    params[_p(prefix,'b_o')] = zero_bias(n_h)
    params[_p(prefix,'b_c')] = zero_bias(n_h)
    
    return params
    
def lstm_layer(tparams, x, mask, prefix):

    n_steps = x.shape[0]
    n_samples = x.shape[1]

    n_h = tparams[_p(prefix,'U_i')].shape[0]

    x_i = tensor.dot(x, tparams[_p(prefix, 'W_i')]) + tparams[_p(prefix, 'b_i')] 
    x_f = tensor.dot(x, tparams[_p(prefix, 'W_f')]) + tparams[_p(prefix, 'b_f')] 
    x_o = tensor.dot(x, tparams[_p(prefix, 'W_o')]) + tparams[_p(prefix, 'b_o')]  
    x_c = tensor.dot(x, tparams[_p(prefix, 'W_c')]) + tparams[_p(prefix, 'b_c')] 

    def _step(m_, xt_i, xt_f, xt_o, xt_c, h_, c_, U_i, U_f, U_o, U_c):
        i = tensor.nnet.sigmoid(tensor.dot(h_, U_i) + xt_i)
        f = tensor.nnet.sigmoid(tensor.dot(h_, U_f) + xt_f)
        o = tensor.nnet.sigmoid(tensor.dot(h_, U_o) + xt_o)
        c = tensor.tanh(tensor.dot(h_, U_c) + xt_c)
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    seqs = [mask, x_i, x_f, x_o, x_c]
    non_seqs = [tparams[_p(prefix, 'U_i')],tparams[_p(prefix, 'U_f')],
				tparams[_p(prefix, 'U_o')],tparams[_p(prefix, 'U_c')]]
    rval, updates = theano.scan(_step,
                                sequences = seqs,
                                outputs_info = [ tensor.alloc(numpy_floatX(0.), n_samples,n_h),
                                                 tensor.alloc(numpy_floatX(0.), n_samples,n_h)],
                                non_sequences = non_seqs,
                                name = _p(prefix, '_layers'),
                                n_steps = n_steps,
                                strict = True)
    # hseq, cseq
    return rval 
