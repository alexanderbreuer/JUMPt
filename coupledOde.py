import torch as tc, numpy as np

def gamma_setter(obj,idx,x):
    obj.gamma[idx] = tc.DoubleTensor(x).to(obj.gamma.device)**.5

class coupledOde(tc.nn.Module):
    def __init__( self, nProteins, etaP, LysConc, ThetaP0, ThetaL0, initGamma ):
        super(coupledOde,self).__init__()
        self.gamma = tc.nn.Parameter(tc.DoubleTensor(initGamma)**.5)
        self.gammaPidx = tc.arange(nProteins)
        self.gammaLidx = nProteins
        self.c = tc.DoubleTensor(etaP/LysConc)
        self.register_parameter('Gamma',self.gamma)
        self.theta0 = tc.DoubleTensor(np.vstack((ThetaP0.reshape((-1,1)),ThetaL0.reshape((-1,1)))))

    def cuda( self ):
        super(coupledOde,self).cuda()
        return self.changeDevice()

    def cpu( self ):
        super(coupledOde,self).cpu()
        return self.changeDevice()
    
    def changeDevice( self ):
        self.c = self.c.to(self.gamma.device)
        self.theta0 = self.theta0.to(self.gamma.device)
        return self
        
    def A( self ):
        A = tc.DoubleTensor().new_zeros((self.gamma.shape[0],self.gamma.shape[0]),device=self.gamma.device)
        A[range(A.shape[0]-1),range(A.shape[1]-1)] = -(self.gamma[:-1]**2)
        A[:-1,-1] = self.gamma[:-1]**2
        A[-1,:-1] = self.c*(self.gamma[:-1]**2)
        A[-1,-1] = -(self.gamma[-1]**2) - (self.c*(self.gamma[:-1]**2)).sum()
        return A

    def forward( self, t ):
        Lambda2,U2 = tc.eig(self.A().detach(),eigenvectors=True)
        Ureal2,Uimag2 = generateU(Lambda2,U2)
        AUreal = tc.mm(self.A(),Ureal2)
        AUimag = tc.mm(self.A(),Uimag2)
        Lambda = tc.DoubleTensor().new_empty(Lambda2.shape,device=Lambda2.device)
        Lambda[:,0] = tc.diag(tc.mm(tc.transpose(AUreal,0,1),Ureal2) +
                              tc.mm(tc.transpose(AUimag,0,1),Uimag2))
        Lambda[:,1] = tc.diag(tc.mm(tc.transpose(AUimag,0,1),Ureal2) -
                              tc.mm(tc.transpose(AUreal,0,1),Uimag2))
        Ureal,Uimag = cdiv(AUreal,AUimag,Lambda[:,0],Lambda[:,1],tc.div,tc.mul)
        
        UrealInv,UimagInv = generateUinv(Ureal,Uimag)
        xreal = tc.mm(UrealInv,self.theta0)
        ximag = tc.mm(UimagInv,self.theta0)
#        t = tc.DoubleTensor().new_tensor(t,device=U2.device)
        yreal,yimag = cmul( tc.exp(t*Lambda[:,0].reshape((-1,1))) * tc.cos(t*Lambda[:,1].reshape((-1,1))),
                            tc.exp(t*Lambda[:,0].reshape((-1,1))) * tc.sin(t*Lambda[:,1].reshape((-1,1))),
                            xreal*tc.DoubleTensor().new_ones((1,t.shape[0]),device=U2.device),
                            ximag*tc.DoubleTensor().new_ones((1,t.shape[0]),device=U2.device), tc.mul )
        Z = tc.mm(Ureal,yreal)

        return Z        

    def setGammaP( self, newGammaP ):
        for i in range(len(newGammaP)):
            gamma_setter(self,self.gammaPidx[i],(newGammaP[i],))
    
    def getGammaP( self ):
        return np.array((self.gamma[self.gammaPidx]**2).clone().detach().cpu())

    def setGammaL( self, newGammaL ):
        gamma_setter(self,self.gammaLidx,newGammaL)

    def getGammaL( self ):
        return np.array((self.gamma[self.gammaLidx]**2).clone().detach().cpu())

    def integrate( self, t ):
        return np.array(self.forward(t).cpu())
    
def generateUinv( Ureal, Uimag ):
    UinvReal = tc.pinverse(Ureal + tc.mm(tc.mm(Uimag,tc.pinverse(Ureal)),Uimag),rcond=1e-9)
    UinvImag = tc.pinverse(Uimag + tc.mm(tc.mm(Ureal,tc.pinverse(Uimag)),Ureal),rcond=1e-9)    
    return UinvReal,UinvImag
    
def generateU( Lambda, U ):
    i = 0
    Ureal = tc.DoubleTensor().new_zeros(U.shape,device=U.device)
    Uimag = tc.DoubleTensor().new_zeros(U.shape,device=U.device)
    while i < Lambda.shape[0]:
        if i+1 < Lambda.shape[0] and tc.abs(Lambda[i,0] - Lambda[i+1,0]) < 1e-17 and tc.abs(Lambda[i,1] + Lambda[i+1,1]) < 1e-17:
            Ureal[:,i] = Ureal[:,i+1] = U[:,i]
            Uimag[:,i] = U[:,i+1]
            Uimag[:,i+1] = -U[:,i+1]
            i += 2
        else:
            Ureal[:,i] = U[:,i]
            i += 1

    return (Ureal,Uimag)

def cmul( areal, aimag, breal, bimag, mop ):
    return (mop(areal,breal),
            mop(areal,bimag) + mop(aimag,breal))

def cdiv( areal, aimag, breal, bimag, dop, mop ):
    return (dop(areal,breal.reshape((1,-1))),mop(aimag,0))
