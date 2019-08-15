import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de

import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de
reader = csv.reader(open('Finaldata_proteins_08-12-2019.csv'))
l = [i for i in reader]
data = np.array(l[1:]).astype(np.double)
t = data[:,0]
ThetaT = data[:,list(range(2,data.shape[1])) + [1]]
reader = csv.reader(open('Finaldata_EtaP_08-12-2019.csv'))

etaP = np.array([i for i in reader][1:]).astype(np.double)
LysConc = 200

n = 1e4

rt,ft,dft = nd.diffFourier( t, ThetaT, samp=n )

ti = nr.randint(0,rt.shape[0],1000)
ThetaH = np.hstack([ft[ti,-1].reshape((-1,1)) - ft[ti,i].reshape((-1,1)) 
                    for i in range(ThetaT.shape[1]-1)])
ThetaA = .05 - ft[ti,2]

A,b,f = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), dft[ti,:-1], dft[ti,-1].reshape((-1,1)), etaP.reshape((-1,))/LysConc )
x = de.solveSparseLstSq( A, b )
