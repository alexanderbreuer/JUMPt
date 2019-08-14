import numpy as np, scipy.sparse as ss, numpy.linalg as nl, scipy.fftpack as sft

def linSys( ThetaHat, ThetaA, dThetaHat, dThetaA, etaP_LysC ):
    """
    Generate one system for a time point.

    ThetaHat := r(Theta_A(t) - Theta_p(t))  (where r is the scalar that we discussed today)
    ThetaHat is num_protiens x 1
    ThetaA is a scalar

    output: A,b for min_x \|Ax - b\|_2^2
    """
    n = ThetaHat.shape[0] + 1
    r = np.hstack((np.arange(n),
                   (n-1)*np.ones(n-1)))
    c = np.hstack((np.arange(n),
                   np.arange(n-1)))
    v = np.hstack((ThetaHat.T,np.ones(1)*ThetaA,-ThetaHat.T*etaP_LysC))
    A = ss.coo_matrix( (v,(r,c)), shape=(n,n) )
    b = np.hstack((dThetaHat.T,np.ones([1]*len(dThetaHat.shape))*dThetaA))
    return (A,np.matrix(b).T)

def lstSqMultiPt( ThetaHat, ThetaA, dThetaHat, dThetaA, etaP_LysC ):
    """
    Generate one system for m time points.

    ThetaHat := r(Theta_A(t) - Theta_p(t))  (where r is the scalar that we discussed today)
    ThetaHat is num_protiens x m (where m is the number of time points)
    ThetaA is m x 1

    output: A,b,funpack for min_x \|Ax - b\|_2^2
            funpack(x) is a function that returns (gamma_p,gamma_a) when given x.
    """
    Abl = [linSys( ThetaHat[i,:], ThetaA[i], dThetaHat[i,:], dThetaA[i], etaP_LysC )
           for i in range(ThetaHat.shape[0])]
    return (ss.vstack([t[0] for t in Abl]),np.vstack([t[1] for t in Abl]),
            lambda x: (x[:-1],x[-1]))

def solveSparseLstSq( A, b ):
    """
    Solve a sparse least squares problem \min_x \|Ax - b\|_2^2

    Uses the normal equations A.TAx = b
    """
    G = (A.T*A).todense()
    return G**-1*(A.T*b)

def diff( fx ):
    """
    Numerically differentiate fx
    """
    FX = sft.rfft(np.hstack((fx.T,fx.T*0)))
    L = 2*np.pi
    for k in range(FX.shape[0]):
        FX[k] *= np.exp(2.*np.pi*1j*k/L)    
    return sft.irfft(FX)
