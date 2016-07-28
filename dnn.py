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

#################
### Class DNN ###
#################
class DNN(object):
    def __init__(self, n_layer_list, name, LR=0.01, x=None, algo=None, activation=T.tanh, clipNorm=0.):
        self.name = name
        self.LR = LR
        self.n_layer_list = n_layer_list

        # Learning_rate from big to small as follows
        # adagrad > adadelta > rmsprop > momentum > nag
        self.adagrad = 'AdaGrad'
        self.adadelta = 'AdaDelta'
        self.rmsprop = 'RMSprop'
        self.momentum = 'Momentum'
        self.nag = 'NAG' #Nesterov Accelerated Gradient

        self.Algo = algo if algo != None else self.adadelta

        self.clipNorm = clipNorm

        n_layer = len(n_layer_list) - 1

        ### DNN Layers Define ###
        x_seq = T.matrix() if x is None else x

        self.hiddenLayerList = []
        for n in range(n_layer):
            name = str(n+1) + '_' + self.name
            d_in = n_layer_list[n]
            d_out = n_layer_list[n+1]

            if n == 0 and n != n_layer-1:
                hiddenLayer = HiddenLayer(x_seq, name, d_in, d_out)
            elif n == 0 and n == n_layer-1:
                hiddenLayer = HiddenLayer(x_seq, name, d_in, d_out,
                                          None)
            elif n == n_layer-1:
                input = self.hiddenLayerList[n-1].output
                hiddenLayer = HiddenLayer(input, name, d_in, d_out,
                                          None)
            else:
                input = self.hiddenLayerList[n-1].output
                hiddenLayer = HiddenLayer(input, name, d_in, d_out)

            self.hiddenLayerList.append(hiddenLayer)

        self.output = self.hiddenLayerList[-1].output
        ### DNN Layers EndDef ###

        ### Parameters ###
        self.params = []
        for hiddenLayer in self.hiddenLayerList:
            self.params += (hiddenLayer.params)

        y_hat = T.matrix()
        
        cost = ((self.output-y_hat)**2).sum() + \
               self.L1(self.params)

        self.cost = cost

        gradients = T.grad(cost, self.params) 
        #gradients = [T.clip(g,-1,1) for g in T.grad(cost,self.params)]

        ### Model Functions ###
        self.train = theano.function(
            inputs=[x_seq,y_hat],
            updates=self.updates(self.params, gradients, self.LR, \
                                 self.Algo, self.clipNorm),
            outputs=[cost],
            allow_input_downcast=True
            )
        self.test = theano.function(
            inputs=[x_seq],
            outputs=[self.output],
            allow_input_downcast=True
            )

    def transform(x):
        output = 0
        for layer in self.hiddenLayerList:
            output = layer.transform(x)
            x = output
        return output

    ### Updates Functions ###
    def updates(self, params, gradients, learning_rate, option=None, clipNorm=0.):
        if clipNorm:
            newGrad = []
            for grad in gradients:
                rate = clipNorm/T.sqrt(T.sum(grad**2))
                newGrad.append(grad*rate)
            gradients = newGrad
        if option == None:
            return self.SGD(params,gradients,learning_rate)
        elif option == self.adagrad:
            return self.AdaGrad(params,gradients,learning_rate)
        elif option == self.adadelta:
            return self.AdaDelta(params,gradients,learning_rate)
        elif option == self.momentum:
            return self.Momentum(params,gradients,learning_rate)
        elif option == self.rmsprop:
            return self.RMSprop(params,gradients,learning_rate)
        elif option == self.nag:
            return self.NAG(params,gradients,learning_rate)
        else:
            print 'ERROR: Updates Function Unavailable'
            exit()

    def SGD(self, params, gradients, learning_rate):
        return [ (p, p - learning_rate*g) 
                 for p,g in izip(params,gradients) ]

    def AdaGrad(self, params, gradients, learning_rate, epsilon=1e-8):
        updates = []
        for p,g in izip(params, gradients):
            acc = theano.shared(p.get_value()*0.)
            acc_new = acc + (g**2)
            scaling = T.sqrt(acc_new + epsilon)
            g = g/scaling
            updates.append((acc, acc_new))
            updates.append((p, p - learning_rate*g))
        return updates

    def Momentum(self, params, gradients, learning_rate, gamma=0.9):
        updates = []
        for p,g in izip(params, gradients):
            last = theano.shared(p.get_value()*0.)
            last_new = gamma * last + learning_rate * g
            updates.append((last, last_new))
            updates.append((p, p - last_new))
        return updates

    def AdaDelta(self, params, gradients, learning_rate, gamma=0.9, \
                 epsilon=1e-8):
        updates = []
        for p,g in izip(params, gradients):
            acc_g = theano.shared(p.get_value()*0.)
            acc_g_new = gamma*acc_g + (1 - gamma)*((g**2))
            acc_theta = theano.shared(p.get_value()*0.)
            scaling = T.sqrt((acc_theta+epsilon)/(acc_g_new+epsilon))
            g = g * scaling
            acc_theta_new = gamma*acc_theta + (1 - gamma)*(g**2)
            updates.append((acc_g, acc_g_new))
            updates.append((acc_theta, acc_theta_new))
            updates.append((p, p - learning_rate * g))
        return updates

    def RMSprop(self, params, gradients, learning_rate, gamma=0.9, \
                epsilon=1e-8):
        updates = []
        for p,g in izip(params, gradients):
            acc = theano.shared(p.get_value()*0.)
            acc_new = gamma*acc + (1 - gamma)*(g**2)
            scaling = T.sqrt(acc_new + epsilon)
            g = g/scaling
            updates.append((acc, acc_new))
            updates.append((p, p - learning_rate*g))
        return updates

    def NAG(self, params, gradients, learning_rate, gamma=0.9):
        updates = []
        for p,g in izip(params,gradients):
            last = theano.shared(p.get_value()*0.)
            last_new = gamma*last - learning_rate*g
            updates.append((last, last_new))
            updates.append((p, p - learning_rate*g + gamma*last_new))
        return updates

    ### L1 L2 Norm ###
    def L1(self, params):
        return T.sum([ abs(x).sum() for x in params ])

    def L2_sqr(self, params):
        return T.sum([ (x**2).sum() for x in params ])

#########################
### Class HiddenLayer ###
#########################
class HiddenLayer(object):
    def __init__(self, input, name, d_in, d_out, activation=T.tanh, W=None, B=None):
        self.name = name
        if W == None:
            W = shared_uniform("W"+self.name, d_out, d_in)
        if B == None:
            B = shared_uniform("B"+self.name, d_out)
        self.W = W
        self.B = B
        self.params = [ self.W, self.B ]
        self.act = activation

        z = T.transpose(T.dot(self.W, input).T + self.B)
        self.output = z if activation is None else activation(z)

    def L1(self):
        return abs(self.W).sum() + abs(self.B).sum()

    def L2_sqr(self):
        return (self.W**2).sum() + (self.B**2).sum()

    def transform(self,x):
        z = T.transpose(T.dot(self.W, x).T + self.B)
        return z if self.act is None else self.act(z)

