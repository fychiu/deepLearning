import theano
import theano.tensor as T
import numpy as np
from itertools import izip

########################
### Useful functions ###
########################
def shared_zeros(name, n_rows, n_cols=None):
    size = (n_rows) if (n_cols is None) else (n_rows, n_cols)
    return theano.shared(np.array(
        np.zeros(shape=size),
        dtype=theano.config.floatX), name=name)


def shared_uniform(name, n_rows, n_cols=None, low=-0.1, high=0.1):
    size = (n_rows) if (n_cols is None) else (n_rows, n_cols)
    return theano.shared(np.array(
        np.random.uniform(low, high, size=size),
        dtype=theano.config.floatX), name=name)


def shared_normal(name, n_rows, n_cols=None, mu=0.0, sigma=0.1):
    size = (n_rows) if (n_cols is None) else (n_rows, n_cols)
    return theano.shared(np.array(
        np.random.normal(mu, sigma, size=size),
        dtype=theano.config.floatX), name=name)


def sigmoid_func(z):
    # Can use hard_sigmoid(z)
    # To automotically change all in program,
    # use optimizer_including=local_hard_sigmoid
    return T.nnet.sigmoid(z)

##################
### Class LSTM ###
##################
class LSTM(object):
    def __init__ (self, d_in, d_out, LR, alpha, grad_bound, x=None, q=None, choices=None, batchSize=1):
        # d_in: dimension of input
        # d_out: dimension of output
        # LR: learning rate
        # alpha: RMSProp alpha
        # grad_bound: gradient clip
        
        self.LR = LR
        self.alpha = alpha
        self.batchSize = batchSize
        
        self.W = shared_uniform("W",d_out,d_in+d_out)
        self.B = shared_uniform("B",d_out)
        self.Wi = shared_uniform("Wi",d_out,d_in+d_out)
        self.Bi = shared_uniform("Bi",d_out)
        self.Wf = shared_uniform("Wf",d_out,d_in+d_out)
        self.Bf = shared_uniform("Bf",d_out)
        self.Wo = shared_uniform("Wo",d_out,d_in+d_out)
        self.Bo = shared_uniform("Bo",d_out)
        
        self.params = self.Wi, self.Bi, self.Wf, self.Bf, \
                        self.W, self.B, self.Wo, self.Bo

        self.batch_grad = []
        for p in self.params:
            self.batch_grad.append(theano.shared(p.get_value()*0.))
        
        ### Step Function ###
        def step(x_t, h_tm1, c_tm1):
            x_h = T.concatenate([h_tm1, x_t],axis=0)
            _inputGate = sigmoid_func( T.dot(self.Wi,x_h) + self.Bi)
            _input = T.tanh( T.dot(self.W, x_h) + self.B)
            _forgetGate = sigmoid_func( T.dot(self.Wf, x_h) + self.Bf)
            c_t = _inputGate * _input + _forgetGate * c_tm1
            _outputGate = sigmoid_func( T.dot(self.Wo, x_h) + self.Bo)
            h_t = _outputGate * T.tanh(c_t)
            return h_t, c_t

        
        x_seq = T.matrix() if x is None else x
        c_0 = shared_zeros('c_0',d_out)
        h_0 = shared_zeros('h_0',d_out)
        
        [self.h_seq, c_seq], self.updates_train = theano.scan( 
            step,
            sequences = x_seq, 
            outputs_info = [h_0, c_0]
            )

        self.q = False if q is None else True
        if self.q:
            q_seq = T.matrix() if q is None else q
            cq_0 = shared_zeros('cq_0',d_out)
            hq_0 = shared_zeros('hq_0',d_out)
            
            [self.hq_seq, cq_seq], self.updates_train = theano.scan( 
                step,
                sequences = q_seq, 
                outputs_info = [hq_0, cq_0]
                )

        self.choices = False if choices is None else True
        if self.choices:
            self.hOpt_seq_list = []
            for i in range(len(choices)):
                opt_seq = T.matrix() if choices is None else choices[i]
                name_c = 'cOpt%d_0' % (i)
                name_h = 'hOpt%d_0' % (i)
                cOpt_0 = shared_zeros(name_c,d_out)
                hOpt_0 = shared_zeros(name_h,d_out)
            
                [hOpt_seq, cOpt_seq],self.updates_train = theano.scan( 
                    step,
                    sequences = q_seq, 
                    outputs_info = [hOpt_0,cOpt_0]
                    )

                self.hOpt_seq_list.append(hOpt_seq)
        
        y_hat = T.vector()
        y_seq = self.h_seq[-1]
        
        ### Cost Function ###
        # SER
        cost = T.sum((y_seq-y_hat)**2)
        # cosine
        #cost = T.sum((y_seq*y_hat)**2)/(T.sum(y_seq**2)*T.sum(h_hat**2))
        
        ### Calculate Gradients ###
        gradients = T.grad(cost, self.params)
        
        self.train = theano.function(
            inputs = [x_seq, y_hat],
            updates = self.rmsprop(self.params, gradients, self.LR, self.batch_grad),
            outputs = [cost],
            allow_input_downcast = True
            )

	self.test = theano.function(
            inputs = [x_seq],
            outputs = y_seq,
            allow_input_downcast = True
            )

        self.output = theano.function(
            inputs = [x_seq],
            outputs = self.h_seq,
            allow_input_downcast = True
            )

    ### RMSProp ###
    # This is copied directly from the Internet
    '''
    def rmsprop(self, parameter, gradient, lr, rho=0.95, epsilon=1e-6):
        updates = []
        for p,g in izip(parameter, gradient):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho*acc + (1-rho) * g ** 2
            scaling = T.sqrt(acc_new + epsilon)
            g = g / scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates
    '''

    def rmsprop(self, parameter, gradient, lr, batch_grad, rho=0.95, epsilon=1e-6):
        updates = []
        index = 0
        for p,g in izip(parameter, gradient):
            batch_grad_new = (batch_grad[index] + g)/self.batchSize
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho*acc + (1-rho) * batch_grad_new ** 2
            scaling = T.sqrt(acc_new + epsilon)
            batch_grad_new = batch_grad_new / scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * batch_grad_new))
            updates.append((batch_grad[index], batch_grad[index]*0.))
            index += 1
        return updates

    def rmsprop_batch(self, parameter, gradient, lr, batch_grad, rho=0.95, epsilon=1e-6):
        updates = []
        index = 0
        for p,g in izip(parameter, gradient):
            batch_grad_new = batch_grad[index] + g
            updates.append((batch_grad[index], batch_grad_new))
            index += 1
        return updates
        
    def train(x_seq, y_hat, backward=False):
        return self.train(x_seq[::-1], y_hat) if backward else self.train(x_seq, y_hat)    

    def test(x_seq, backward=False):
        return self.test(x_seq)

    def output_encoding(self): #backward=False):
        outputList = []
        outputList.append(self.h_seq)
        if self.q is True:
            outputList.append(self.hq_seq)
        if self.choices is True:
            outputList.extend(self.hOpt_seq_list)
        return outputList

