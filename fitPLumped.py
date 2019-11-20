import csv, numpy as np, torch as tc, coupledOde as co, scipy.optimize as so, numpy.random as nr, parseH as ph, h5py, time, sys, generateUnobs as gu

t,ft,header = ph.parseH( 'Min3data_2iso_1105_incl_UIP.csv' )
LysConc = 200
nuip = 128

i = int(sys.argv[1])+1
f = None
while f is None:
    try:
        f = h5py.File( 'Min3data_2iso_1105_Lys.h5', 'r' )
    except Exception:
        time.sleep(.1)

fL = np.array(f['Lys/f'])
dfL = np.array(f['Lys/df'])
tfL = np.array(f['Lys/t'])
f.close()

noFun = gu.objectiveFunction( t[i], ft[i], tfL, dfL ) 
result =  so.minimize_scalar( noFun, bounds=[0,20], 
                              method='bounded', options={'print':3,'disp':3,'maxiter':2000} )
fP = gu.generateSys( tfL, dfL, result.x )
dfP = result.x*(fL - fP)

of = None
while of is None:
    try:
        of = h5py.File( 'Min3data_2iso_1105_Lys.h5', 'r+' )
    except Exception:
        time.sleep(.1)

of.create_dataset( 'p{}/t'.format(i), data=tfL )
of.create_dataset( 'P{}/tf'.format(i), data=fP )
of.create_dataset( 'P{}/tdf'.format(i), data=dfP )
of.close()
