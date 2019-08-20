import torch as tc, numpy as np

class MultiExp(tc.nn.Module):
    def __init__( self, nExp ):
        super(MultiExp,self).__init__()
        self.nExp = nExp
        self.Lambda = tc.nn.Parameter(tc.DoubleTensor().new_empty((nExp,)))
        self.c = tc.nn.Parameter(tc.DoubleTensor().new_empty((nExp,)))
        tc.nn.init.normal_(self.c)
        tc.nn.init.uniform_(self.Lambda)
        self.register_parameter('lambda',self.Lambda)
        self.register_parameter('c',self.c)
        
    def forward( self, t ):
        return (self.c*tc.exp(-tc.abs(self.Lambda)*t.reshape((-1,1)))).sum(1)

def ft( Lambda, C, t ):
    return np.multiply(np.exp(-Lambda.reshape(list(Lambda.shape)+[1])*t.reshape((1,1,-1))),C.reshape(list(C.shape)+[1])).sum(0)

def dft( Lambda, C, t ):
    return np.multiply(np.multiply(-Lambda.reshape(list(Lambda.shape)+[1]),
                                   np.exp(-Lambda.reshape(list(Lambda.shape)+[1])*t.reshape((1,1,-1)))),
                       C.reshape(list(C.shape)+[1])).sum(0)
