import csv, numpy as np, scipy.optimize as so, numpy.random as nr, numDiff as nd, de

reader = csv.reader(open('test_data2.csv'))
l = [i for i in reader]
data = np.array(l[1:]).astype(np.double)

t = data[:,0]
ThetaT = data[:,(2,3,1)]
etaP = np.array((3,6))
LysConc = 200
n = 1e5

rt,dt = nd.diffFourier( t, ThetaT, samp=n )
ti = (t*n).astype(np.uint64)[1:-1]
ThetaH = np.hstack([ThetaT[1:-1,2].reshape((-1,1)) - ThetaT[1:-1,0].reshape((-1,1)),ThetaT[1:-1,2].reshape((-1,1)) - ThetaT[1:-1,1].reshape((-1,1))])
ThetaA = .05 - ThetaT[1:-1,2]

A,b,f = de.lstSqMultiPt( ThetaH, ThetaA.reshape((-1,1)), dt[ti,:2], dt[ti,2].reshape((-1,1)), etaP/LysConc )
x = de.solveSparseLstSq( A, b )
