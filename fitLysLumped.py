import csv, numpy as np, torch as tc, coupledOde as co, scipy.optimize as so, numpy.random as nr, parseH as ph, h5py

t,ft,header = ph.parseH( 'Min3data_2iso_1105_incl_UIP.csv' )
LysConc = 200
nuip = 160

ThetaT = np.hstack((np.ones((ft[0].shape[0],nuip)),ft[0].reshape((-1,1))))
result =  so.minimize( co.createObjFunction(np.ones(nuip),LysConc,.05,t[0],ThetaT,mapper=lambda x: x[-1,:]), nr.uniform(0,1,nuip+1),
                       method='trust-constr', options={'verbose':3,'disp':True,'maxiter':2**20} )

nch = 2048
tch = np.arange(0,32+1e-2,1e-2)

reg = co.coupledOde( nuip, np.ones(nuip), LysConc, ThetaT.T[:-1,0], np.array(ThetaT.T[-1,0]), np.array([.05]), result.x )
fL = reg.forward(tc.DoubleTensor(tch))
dfL = tc.mm(reg.A(),tc.cat((fL,tc.DoubleTensor().new_ones((1,tch.shape[0]))*.05),0))[-2,:]

of = h5py.File( 'Min3data_2iso_1105_Lys.h5', 'w' )
of.create_dataset( 'Lys/t', data=tch )
of.create_dataset( 'Lys/f', data=fL.detach().numpy() )
of.create_dataset( 'Lys/df', data=dfL.detach().numpy() )
of.close()
