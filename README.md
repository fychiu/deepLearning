# deepLearning
The codes are written for projects and practice






Modules:

dnn.py:

    Inludes layer module and dnn module
    
    The dnn module has several optimization functions:
    
        Reference(I): http://sebastianruder.com/optimizing-gradient-descent/
        The websites states the function and theory of each optimization algorithm
        
        Reference(II): http://lasagne.readthedocs.io/en/latest/modules/updates.html
        The Lasagne Library gives me the examples of implementation
        Also the websites briefly describe the mathematic equations of algorithms
        
        
    HiddenLayer:
    
        Reference(I): http://deeplearning.net/tutorial/mlp.html
        The example of theano let me know more clear about how a module of DNN
        should be implemented
    

attenBased.py:

    Implement the attention-based mechanism structure of Google's paper
    Teaching Machine ot Read and Comprehend
        http://arxiv.org/abs/1506.03340
    
    The file will return three models:
    1. train_model: 
        a) input (1) a sequence of document word vector, 
                 (2) a sequence of question word vector, and 
                 (3) the correct word vector.
        b) train model
        
    2. test_model:
        a) input (1) a sequence of document word vector, 
                 (2) a sequence of question word vector, and 
        b) output a word vector
        
    3. testAns_model:
        a) input (1) a sequence of document word vector, 
                 (2) a sequence of question word vector, and
                 (3) four sequences of option word vector
                 so totally, 6 inputs
        b) output the index of answer my model predicts
            Ex: when answer is predicted as B option, the output is 1
                                            A option, the output is 0
    

Other references:
    
    Chainer Neural Network Framework:
        http://docs.chainer.org/en/stable/index.html
