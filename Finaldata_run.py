import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de, h5py, multiExp as me

f = h5py.File('Finaldata_proteins_Lambda_C.h5','r')
Lambda = np.array(f['Lambda'])
C = np.array(f['Lambda'])

reader = csv.reader(open('Finaldata_EtaP_08-12-2019.csv'))
etaP = np.array([i for i in reader][1:]).astype(np.double)
LysConc = 200 

n = 1e5

t = np.linspace(0,32,int(n))
ti = nr.randint(0,t.shape[0],1000)
t = t[tidx]
f = me.ft(Lambda,C,t).T
df = me.dft(Lambda,C,t).T

ThetaH = np.hstack([f[:,0].reshape((-1,1)) - f[:,i].reshape((-1,1)) 
                    for i in range(1,f.shape[1])])
ThetaA = .05 - f[:,0]

A,b = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), df[:,1:], df[:,0].reshape((-1,1)), etaP.reshape((-1,))/LysConc )[:2]
x = de.solveSparseLstSq( A, b )
