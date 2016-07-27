import theano
import theano.tensor as T
import numpy as np
from myLSTM import *

def buildAttenBased(d_in, d_out, LR, alpha):
    # fw_lstm_doc:  forward LSTM of document encoding
    # bw_lstm_doc: backward LSTM of document encoding
    # fw_lstm_que:  forward LSTM of question encoding
    # bw_lstm_que: backward LSTM of question encoding
    # y: words encoding, total t words, so t vectors
    # u: question encoding, only one vector

    x_seq = T.matrix('x_seq')
    q_seq = T.matrix('q_seq')

    fw_lstm_doc = LSTM(d_in, d_out, LR, alpha, 0, x_seq)
    bw_lstm_doc = LSTM(d_in, d_out, LR, alpha, 0, x_seq[::-1])
    fw_lstm_que = LSTM(d_in, d_out, LR, alpha, 0, q_seq)
    bw_lstm_que = LSTM(d_in, d_out, LR, alpha, 0, q_seq[::-1])

    y = T.concatenate([
            fw_lstm_doc.output_encoding(),
            bw_lstm_doc.output_encoding()[::-1] 
            ],
            axis=1)
    u = T.concatenate([ 
            [fw_lstm_que.output_encoding()[-1]],
            [bw_lstm_que.output_encoding()[-1]] 
            ],
            axis=1)

    Wym = shared_uniform("Wym", 2*d_out, 2*d_out)
    Wum = shared_uniform("Wum", 2*d_out, 2*d_out)
    wms = shared_uniform("wms", 2*d_out)
    Wrg = shared_uniform("Wrg", 2*d_out, 2*d_out)
    Wug = shared_uniform("Wug", 2*d_out, 2*d_out)
    
    yT = T.transpose(y)
    uT = T.transpose(u)
    um = T.dot(Wum,uT)
    m = T.tanh( T.dot(Wym, yT) + um ) 
    
    # s is a vector (t,)
    mT = T.transpose(m)
    s = T.exp( T.sum(wms*mT, axis = 1) )
    s = s/T.sqrt(T.sum(s**2))
    
    # r is a vector (2*d_out,)
    # ug is (1, 2*d_out)
    # g is a vector (2*d_out,)
    r = T.dot(T.transpose(y), s)
    ug = T.transpose(T.dot(Wug,uT))
    g = T.sum( T.tanh( T.dot(Wrg,r) + ug ), axis = 0)

    g_hat = T.vector('g_hat')

    ### Cost Function ###
    cost = T.sum((g-g_hat)**2)

    params = [Wym, Wum, wms, Wrg, Wug]
    params.extend(fw_lstm_doc.params)
    params.extend(bw_lstm_doc.params)
    params.extend(fw_lstm_que.params)
    params.extend(bw_lstm_que.params)
    
    ### Calclate Gradients ###
    gradients = T.grad(cost, params)

    ### Model Functions ###
    train_model = theano.function(
        inputs = [x_seq, q_seq, g_hat],
        updates = fw_lstm_doc.rmsprop(params, gradients, LR),
        outputs = [cost],
        allow_input_downcast = True
        )

    test_model = theano.function(
        inputs = [x_seq, q_seq],
        outputs = g,
        allow_input_downcast = True
        )

    A = T.vector('A_Opt')
    B = T.vector('B_Opt')
    C = T.vector('C_Opt')
    D = T.vector('D_Opt')

    ser = [ T.sum((g-A)**2),T.sum((g-B)**2),
            T.sum((g-C)**2),T.sum((g-D)**2) ]

    opt = T.argmin(ser)
    testAns_model = theano.function(
        inputs = [x_seq, q_seq, A, B, C, D],
        outputs = opt,
        allow_input_downcast = True
        )
        
    return train_model, test_model, testAns_model
