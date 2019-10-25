import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de, h5py, multiExp as me, numpy.linalg as nl, torch as tc, coupledOde as co, parseH as ph, scipy.sparse as ss, control, scipy.interpolate as si

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


A,b = de.lstSqMultiPt( ThetaH[:,:], ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/206, t, .0 )[:2]
B = de.lppBoltOn(ThetaH.shape[0],A.shape[1],t,.0)
M = ss.hstack((A,B)).tocsr()
G = (M.T*M).todense()
U,S,Vt = nl.svd(G)
x = Vt[:-1,:].T*(np.diagflat(S[:-1]/(S[:-1]**2+0))*(U[:,:-1].T*(M.T*b)))

reg = co.coupledOde( Lambda.shape[1]-1, etaP.reshape((-1,)), 206, np.ones(Lambda.shape[1]-1), np.ones(1), np.ones(1)*.05, np.array(x[:A.shape[1],:]).reshape((-1,)) )
sys = control.ss(reg.A().detach(),np.eye(reg.A().shape[0]),np.eye(reg.A().shape[0]),np.zeros(reg.A().shape))
perm = t.argsort()
ut = np.vstack(([[0]],np.array(x[A.shape[1]:,:])[perm]))
tp = np.hstack(([0],t[perm]))
Uv = np.zeros((reg.A().shape[0],500))
Uv[-2,:] = si.UnivariateSpline(tp,ut.reshape((-1,)),s=10)(np.linspace(0,31,500))


x0 = np.ones((2236,1))
x0[-1] = 0.05
rv = control.forced_response(sys,np.linspace(1,31,500),X0=x0,U=Uv)

reft,refft,header = ph.parseH( 'Min4data_2iso_0919.csv', do_squeeze=False )
reft = np.vstack([x.reshape((1,-1)) for x in reft])
refft = np.vstack([x.reshape((1,-1)) for x in refft])
data = refft[:,list(range(1,refft.shape[1]))+[0]]
mask = (1 - np.isnan(data)).nonzero()
residual = (rv[1][:-1,:].T - rv[0])[(0,16,31,

def optLysConc():
    def noFun( lc ):
        A,b = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/lc )[:2]
        x = de.solveSparseLstSq( A, b, W=W )
        return nl.norm(A*x - b)**2
    return minimize_scalar( noFun, LysConc, options={'disp':3} ),
        
