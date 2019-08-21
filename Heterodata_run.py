import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de, h5py, multiExp as me, parseH as ph, torch as tc, numpy.linalg as nl, coupledOde as co

f = h5py.File('Heterodata_Lambda_C.h5','r')
Lambda = np.abs(np.array(f['Lambda']))
C = np.array(f['C'])

reader = csv.reader(open('EtaP_2iso.csv'))
etaP = np.array([i for i in reader][1:]).astype(np.double)
LysConc = 200 

n = 1e5

t = np.linspace(0,32,int(n))
tidx = nr.randint(0,t.shape[0],1000)
t = t[tidx]
f = me.ft(Lambda,C,t).T
df = me.dft(Lambda,C,t).T

ThetaH = np.hstack([f[:,0].reshape((-1,1)) - f[:,i].reshape((-1,1)) 
                    for i in range(1,f.shape[1])])
ThetaA = .05 - f[:,0]

A,b = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/LysConc )[:2]
x = de.solveSparseLstSq( A, b )

G = np.array((A.T*A).todense())
ATb = np.array(A.T*b)[:,0]
res = so.minimize( lambda x: nl.norm(np.dot(G,x) - ATb)**2, np.array(np.abs(x)).reshape((-1,)), jac=lambda x: np.dot(G,x) - ATb, bounds=[(0,None)]*G.shape[0], method='SLSQP', options={'iprint':3,'disp':True,'maxiter':25} )
reg = co.coupledOde( Lambda.shape[1]-1, etaP.reshape((-1,)), 200, np.ones(Lambda.shape[1]-1), np.ones(1), np.ones(1)*.05, np.array(res.x).reshape((-1,)) )

reft,refft,header = ph.parseH( 'Heterodata_2iso.csv', do_squeeze=False )
reft = np.vstack([x.reshape((1,-1)) for x in reft])
refft = np.vstack([x.reshape((1,-1)) for x in refft])
data = refft[:,list(range(1,refft.shape[1]))+[1]]
mask = (1 - np.isnan(data)).nonzero()
residual = reg.forward(tc.DoubleTensor(reft[:,0])).detach().numpy().T - data
flatResidual = residual[mask]
