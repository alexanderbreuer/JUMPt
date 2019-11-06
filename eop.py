import numpy as np, scipy.interpolate as si, numpy.fft as nft, h5py, parseH as ph, scipy.optimize as so, numpy.linalg as nl, numpy.random as nr

class GammaOptimizer:
    def __init__( self, reg ):
        self.reg = reg

    def __call__( self, g ):
        self.reg.gamma = g
        self.reg.optimize()
        dx = (self.reg.evaluate(self.reg.lbx) - self.reg.evaluate(self.reg.lbx-1e-6))/1e-6
        return nl.norm(dx*(dx > 0),1)

    
class RBFRidgeRegressor:
    def __init__( self, x, fx, lbx, lbfx, initGamma, initC, tau, bias ):
        self.x = np.vstack((x.reshape((-1,1)),
                            np.array([x.max() + (x[1] - x[0]),x.max() + 2*(x[1] - x[0])]).reshape((-1,1))))
        self.fx = fx.reshape((-1,1))
        self.lbx = lbx
        self.lbfx = lbfx
        self.gamma = initGamma
        self.tau = tau
        self.c = initC
        self.lbx = lbx
        self.bias = bias.reshape((-1,1))
        self.A = (self.x[:x.shape[0]] - self.x.T)**2

        self.constraints = []
        if not isinstance(lbfx,type(None)):
            assert not isinstance(lbx,type(None))
            self.constraints.append(so.NonlinearConstraint(lambda z: self.evaluate(lbx,gamma=self.gamma,w=z) - lbfx,0,np.inf))
        
        self.z0 = np.empty(self.x.shape[0])
        self.z0[:] = np.array(nl.pinv(np.matrix(self.bias*np.exp(-self.gamma*self.A)),self.c)*(self.bias*self.fx)).reshape((-1,))
        self.w = self.z0
        self.optimize()

    def optimize( self ):
        res = so.minimize( self.loss, self.w, constraints=self.constraints, jac='2-point', hess=so.BFGS(), method='trust-constr', options={'verbose':1,'maxiter':1024,'xtol':1e-8} )
        self.w = res.x

    def evaluate( self, z, gamma=None, w=None ):
        if isinstance(w,type(None)):
            w_ = self.w
        else:
            w_ = w

        if gamma != None:
            return np.dot(np.exp(-gamma*(z.reshape((-1,1)) - self.x.T)**2),w_)
        else:
            return np.dot(np.exp(-self.gamma*(z.reshape((-1,1)) - self.x.T)**2),w_)

    def loss( self, z ):
        gamma = self.gamma
        c = self.c
        dx = (self.evaluate(self.lbx,gamma=gamma,w=z) - self.evaluate(self.lbx-1e-6,gamma=gamma,w=z))/1e-6
        dxmask = dx > 0
        return nl.norm(self.bias*(np.dot(np.exp(-gamma*self.A),z.reshape((-1,1))) - self.fx))**2 + self.c*nl.norm(z)**2 + self.tau*nl.norm(dx*dxmask,1)
        
class ExponentialOfPolynomial:
    def __init__( self, csvfile, s=10, k=2 ):
        self.t,self.ft,header = ph.parseH(csvfile)
        for i in range(1,len(self.t)):
            if self.t[0][-1] not in self.t[i]:
                self.t[i] = np.hstack((self.t[i],[self.t[0][-1]]))
                self.ft[i] = np.hstack((self.ft[i],[self.ft[0][-1]]))

        self.pil = [i
                    for i in range(len(self.t))]
        
    def f( self, x ):
        return np.exp(np.hstack([p(x).reshape((-1,1)) for p in self.pil]))
    
    def df( self, x, delta=1e-6 ):
        return (self.f(x) - self.f(x-delta))/delta
    
if __name__ == '__main__':
    import sys
    if '--bootstrap' in sys.argv:
        sys.argv.pop(sys.argv.index('--bootstrap'))
        bootstrap = True
    elif '--nprot' in sys.argv:
        sys.argv.pop(sys.argv.index('--nprot'))
        bootstrap = False
        nprot = True
    else:
        bootstrap = False
        nprot = False

    t,ft,header = ph.parseH(sys.argv[1])
    rl = []
    tt = nr.uniform(min([tv.min() for tv in t]),2*max([tv.max() for tv in t]),256)
    tt.sort()
    if bootstrap:
        of = h5py.File( sys.argv[2], 'w' )
        of.create_dataset( 'tt', data=tt )
        of.create_dataset( 'tft', data=np.zeros(tt.shape[0]) )
        of.close()
    elif nprot:
        print(len(t))
    else:    
        i = int(sys.argv[2])
        ihf = h5py.File( sys.argv[3], 'r' )
        tt = np.array(ihf['tt']).reshape((-1,))
        tft = np.array(ihf['tft']).reshape((-1,))

        reg = RBFRidgeRegressor( t[i], ft[i], tt, tft, 1e-4, 1e-4, 1e-5, np.array([10] + ([1])*(ft[i].shape[0]-1)) )
        gopt = GammaOptimizer(reg)
        so.minimize_scalar( gopt, bounds=[1e-5,1e-2], method='bounded', options={'disp':3} )

        n = 2048
        ct = ((np.cos(np.pi*(((2*np.arange(1,n+1)) - 1)/4096))) + 1)*16
        ct.sort()

        of = h5py.File( sys.argv[4], 'w' )
        of.create_dataset( 'ft', data=gopt.reg.evaluate(ct) )
        of.create_dataset( 'tft', data=gopt.reg.evaluate(tt) )
        of.create_dataset( 'dft', data=(gopt.reg.evaluate(ct) - gopt.reg.evaluate(ct-1e-6))/1e-6 )
        of.create_dataset( 't', data=ct )
        of.create_dataset( 'tt', data=tt )
        of.close()
