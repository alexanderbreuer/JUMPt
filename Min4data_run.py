import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de, h5py, multiExp as me, numpy.linalg as nl, torch as tc, coupledOde as co, parseH as ph, scipy.sparse as ss

f = h5py.File('Min4data_2iso_0910_C.h5','r')
Lambda = np.abs(np.array(f['Lambda']))
C = np.array(f['C'])

reader = csv.reader(open('Min4data_2iso_EtaP_0910.csv'))
etaP = np.array([i for i in reader][1:]).astype(np.double)
LysConc = 200 

n = 1e5

t = np.array(f['td'])
f = me.ft(Lambda,C,t).T
df = me.dft(Lambda,C,t).T

ThetaH = np.hstack([f[:,0].reshape((-1,1)) - f[:,i].reshape((-1,1)) 
                    for i in range(1,f.shape[1])])
ThetaA = .05 - f[:,0]

def get_x( lc ):
    A,b = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/lc )[:2]
    W = ss.identity(A.shape[0]).tocsr()
#    W[-1,-1] = 1e4
    return A,b,W,de.solveSparseLstSq( A, b, W=W )

def noFun( lc ):
    A,b,W,x = get_x(lc)
    return nl.norm(W*(A*x - b))**2

val = so.minimize_scalar(noFun,bounds=(100,400),method='bounded',options={'disp':3})
A,b,W,x = get_x(val.x)

G = np.array((A.T*W*A).todense())
ATb = np.array(A.T*W*b)[:,0]
res = so.minimize( lambda x: nl.norm(np.dot(G,x) - ATb)**2, np.array(np.abs(x)).reshape((-1,)), jac=lambda x: np.dot(G,x) - ATb, bounds=[(0,None)]*G.shape[0], method='SLSQP', options={'iprint':3,'disp':True,'maxiter':25} )
reg = co.coupledOde( Lambda.shape[1]-1, etaP.reshape((-1,)), LysConc, np.ones(Lambda.shape[1]-1), np.ones(1), np.ones(1)*.05, np.array(res.x).reshape((-1,)) )

reft,refft,header = ph.parseH( 'Min4data_2iso_0919.csv', do_squeeze=False )
reft = np.vstack([x.reshape((1,-1)) for x in reft])
refft = np.vstack([x.reshape((1,-1)) for x in refft])
data = refft[:,list(range(1,refft.shape[1]))+[0]]
mask = (1 - np.isnan(data)).nonzero()
residual = reg.forward(tc.DoubleTensor(reft[:,0])).detach().numpy().T - data
flatResidual = residual[mask]

def optLysConc():
    def noFun( lc ):
        A,b = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/lc )[:2]
        x = de.solveSparseLstSq( A, b, W=W )
        return nl.norm(A*x - b)**2
    return minimize_scalar( noFun, LysConc, options={'disp':3} ),
        
