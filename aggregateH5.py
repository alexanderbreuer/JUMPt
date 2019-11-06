import h5py, sys, numpy as np

of = h5py.File( sys.argv[-1], 'w' )
f = h5py.File( sys.argv[1], 'r' )
keys = list(f)
f.close()
fl = [h5py.File(f,'r') for f in sys.argv[1:-1]]
for k in keys:
    of.create_dataset( k, data=np.hstack([np.array(f)[k]).reshape((-1,1)) for f in fl]) )

of.close()
