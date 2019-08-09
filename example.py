import csv, numpy as np, torch as tc, coupledOde as co, scipy.optimize as so, numpy.random as nr

reader = csv.reader(open('test_data2.csv'))
l = [i for i in reader]
data = np.array(l[1:]).astype(np.double)

t = data[:,0]
ThetaT = data[:,(2,3,1)]
etaP = np.array((3,6))
LysConc = 200

result =  so.minimize( co.createObjFunction(etaP,LysConc,.05,t,ThetaT), nr.uniform(0,1,3),
                       method='SLSQP', options={'iprint':3,'disp':True,'maxiter':20000} )

