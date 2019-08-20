import scipy.io as sio, parseH as ph, tempfile, os.path, shutil, sys, h5py, csv, numpy as np

heterogeneous = False
while '--heterogeneous' in sys.argv:
    heterogenous = True
    sys.argv.pop(sys.argv.index('--heterogeneous'))
    
if heterogeneous == True:
    t,ft,header = ph.parseH(sys.argv[1])
else: 
    reader = csv.reader(open(sys.argv[1]))
    lines = list(reader)
    header = lines[0]
    data = np.array(lines[1:]).astype(np.double)
    ft = [data[:,i] for i in range(1,data.shape[1])]
    t = [data[:,0] for i in range(len(ft))]

path = tempfile.mkdtemp()
for s in ['adaex.m', 'varpro.m']:
    shutil.copy(os.path.join('matlab',s),path)

mtemplate = open('matlab/varpro_example.m').read()
for i in range(len(t)):
    sio.savemat(os.path.join(path,'inputs_{}.mat'.format(i)),{'t':t[i],'ft':ft[i]})
    open(os.path.join(path,'mscript_{}.m'.format(i)),'w').write(mtemplate.replace('INPUT','inputs_{}.mat'.format(i)).replace('OUTPUT','outputs_{}.mat'.format(i)))

open(os.path.join(path,'wrapper.m'),'w').write(str.join(';\n',['run mscript_{}.m'.format(i) for i in range(len(t))])+';\nexit;\n')
os.system('cd {} && matlab -nojvm -nodisplay -r "run {}"'.format(path,'wrapper.m'))

Lambda = np.empty((5,len(t)))
C = np.empty((5,len(t)))
for i in range(len(t)):
  d = sio.loadmat(os.path.join(path,'outputs_{}.mat'.format(i)))
  Lambda[:,i] = d['alpha'].reshape((-1,))
  C[:,i] = d['c'].reshape((-1,))

f = h5py.File( sys.argv[2], 'w' )
f.create_dataset( 'C', data=C )
f.create_dataset( 'Lambda', data=Lambda )
f.attrs['colMap'] = str.join( '\n', header )
f.close()
shutil.rmtree(path)
