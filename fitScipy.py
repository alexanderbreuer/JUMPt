import numpy.linalg as nl, numpy as np, numpy.random as nr, parseH as ph, scipy.optimize as so, sys, h5py

heterogeneous = False
while '--heterogeneous' in sys.argv:
    heterogeneous = True
    sys.argv.pop(sys.argv.index('--heterogeneous'))

nexp = 3
for i in range(len(sys.argv)):
    if '--nexp' in sys.argv[i]:
        k = sys.argv.pop(i)
        nexp = int(k.split('=')[1])
        break
    
if heterogeneous == True:
    t,ft,header = ph.parseH(sys.argv[1])
else: 
    reader = csv.reader(open(sys.argv[1]))
    lines = list(reader)
    header = lines[0]
    data = np.array(lines[1:]).astype(np.double)
    ft = [data[:,i] for i in range(1,data.shape[1])]
    t = [data[:,0] for i in range(len(ft))]


resl = []
for i in range(len(t)):
    nofun = lambda z: (np.abs(z[np.arange(0,z.shape[0],2)])*np.exp(-np.abs(z[np.arange(1,z.shape[0],2)])*t[i].reshape((-1,1)))).sum(1)
    resl.append( so.minimize( lambda z: 10*(nofun(z)[0] - ft[i][0])**2 + nl.norm(nofun(z) - ft[i])**2 + 1e-2*nl.norm(z[np.arange(0,z.shape[0],2)])**2, nr.uniform(0,1,nexp*2), method='SLSQP', options={'iprint':1,'disp':True,'maxiter':10000,'tol':1e-16} ) )


of = h5py.File( sys.argv[2], 'w' )
of.create_dataset( 'C', data=np.abs(np.hstack([res.x[np.arange(0,res.x.shape[0],2)].reshape((-1,1)) for res in resl])) )
of.create_dataset( 'Lambda', data=np.abs(np.hstack([res.x[np.arange(1,res.x.shape[0],2)].reshape((-1,1)) for res in resl])) )
of.close()
