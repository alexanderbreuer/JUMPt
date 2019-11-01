import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de, h5py, multiExp as me, numpy.linalg as nl, torch as tc, coupledOde as co, parseH as ph, scipy.sparse as ss, control, scipy.interpolate as si

f = h5py.File('Min4data_2iso_0910_C.h5','r')
Lambda = np.abs(np.array(f['Lambda']))
C = np.array(f['C'])

reader = csv.reader(open('Min4data_2iso_EtaP_0910.csv'))
etaP = np.array([i for i in reader][1:]).astype(np.double)
LysConc = 200 

n = 2048

t = ((np.cos(np.pi*(((2*np.arange(1,n+1)) - 1)/4096))) + 1)*16
t.sort()
f = me.ft(Lambda,C,t).T
df = me.dft(Lambda,C,t).T

ThetaH = np.hstack([f[:,0].reshape((-1,1)) - f[:,i].reshape((-1,1)) 
                    for i in range(1,f.shape[1])])
ThetaA = -f[:,0]

A,b = de.lstSqMultiPt( ThetaH[:,:], ThetaA.reshape((-1,1)), df[:,1:], (df[:,0]-.05).reshape((-1,1)), etaP.reshape((-1,))/206, t, .0125 )[:2]
B = de.lppBoltOn(ThetaH.shape[0],A.shape[1],t,.0)
M = ss.hstack((A,B)).tocsr()
# preconditioning with Jacobi preconditioner
D = ss.dia_matrix( (1./np.power(M.multiply(M).sum(0),.5),0), (M.shape[1],M.shape[1]) )
G = (D*M.T*M*D).todense()
# Tikhonov matrix for regularization
T = ss.dia_matrix( (np.hstack((np.zeros(A.shape[1]),np.ones(M.shape[1] - A.shape[1]))),0), G.shape )
gammaUt = D*((G + T*1e-15)**-1*D*(M.T*b))

reg = co.coupledOde( Lambda.shape[1]-1, etaP.reshape((-1,)), 206, np.ones(Lambda.shape[1]-1), np.ones(1), np.ones(1)*.05, np.array(gammaUt[:A.shape[1]]).reshape((-1,)) )
sys = control.ss(reg.A().detach(),np.eye(reg.A().shape[0]),np.eye(reg.A().shape[0]),np.zeros(reg.A().shape))
ut = np.vstack(([[0]],np.array(gammaUt[A.shape[1]:]).reshape((-1,1))))
tp = np.hstack(([0],t))
Uv = np.zeros((reg.A().shape[0],641))
Uv[-2,:] = si.UnivariateSpline(tp,ut.reshape((-1,)),s=10)(np.arange(0,32+5e-2,5e-2))

x0 = np.ones((2236,1))
x0[-1] = 0.05
rv = control.forced_response(sys,np.arange(0,32+5e-2,5e-2),X0=x0,U=Uv)

reft,refft,header = ph.parseH( 'Min4data_2iso_0919.csv', do_squeeze=False )
reft = np.vstack([x.reshape((1,-1)) for x in reft])
refft = np.vstack([x.reshape((1,-1)) for x in refft])
residual = rv[1][:-1,(np.power( 2, [0,2,3,4,5] )/5e-2).astype(np.int)] - refft[:,list(range(1,2235))+[0]].T
mask = (1 - np.isnan(residual)).nonzero()
flatResidual = residual[mask]

def optLysConc():
    def noFun( lc ):
        A,b = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/lc )[:2]
        x = de.solveSparseLstSq( A, b, W=W )
        return nl.norm(A*x - b)**2
    return minimize_scalar( noFun, LysConc, options={'disp':3} ),
        
