import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de

reader = csv.reader(open('test_data2.csv'))
l = [i for i in reader]
data = np.array(l[1:]).astype(np.double)

t = data[:,0]
ThetaT = data[:,(2,3,1)]
etaP = np.array((3,6))
LysConc = 200
n = 1e4

rt,ft,dft = nd.diffFourier( t, ThetaT, samp=n )
ti = nr.randint(0,rt.shape[0],1000)
ThetaH = np.hstack([ft[ti,-1].reshape((-1,1)) - ft[ti,i].reshape((-1,1)) 
                    for i in range(ThetaT.shape[1]-1)])
ThetaA = .05 - ft[ti,-1]

A,b,f = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), dft[ti,:2], dft[ti,2].reshape((-1,1)), etaP/LysConc )
x = de.solveSparseLstSq( A, b )
