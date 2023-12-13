# This package contains Easyspin functions that have been ported from Matlab to python.

# Load required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags,lil_matrix
from scipy.sparse.linalg import spsolve

from scipy import constants

def main():
    print('This package contains functions that are supplied by the Easyspin software package')

def mhz2mt(x_MHz,g=abs(constants.value(u'electron g factor'))):
    # This function converts MHz to mT.
    return x_MHz/g*(1e6*constants.h/constants.value(u'Bohr magneton')/1e-3)

def mt2mhz(x_mT,g=abs(constants.value(u'electron g factor'))):
    # This function converts mT to MHz.
    return x_mT*g*(constants.value(u'Bohr magneton')*1e-3/constants.h/1e6)

def t2lor(x_mT):
    # This function converts the first derivative peak-to-peak linewidth to T2
    gamma = constants.value(u'Bohr magneton')/constants.hbar*abs(constants.value(u'electron g factor'))
    return 2/gamma/np.sqrt(3)/(x_mT*1e-3)

def blochsteady(g,T1,T2,DeltaB0,B1,ModAmp,ModFreq,nPoints=0):
    # this function calculates steady state solutions of Bloch's
    # equations with sinusoidal modulation according to
    # M. Tseytlin, G. R. Eaton, S. S. Eaton, Appl. Magn. Reson. 2015,
    # 44, 1373-1379.
    # It is based on the Easyspin function blochsteady

    # The input variables are
    # [g] = 1
    # [T1] = [T2] = us
    # [DeltaB0] = mT
    # [B1] = mT
    # [ModAmp] = mT
    # [ModFreq] = kHz
    # [nPoints] = 1, optional

    # Convert inputs to SI units
    T1 *= 1e-6 # s
    T2 *= 1e-6 # s
    DeltaB0 *= 1e-3 # T
    B1 *= 1e-3 # T
    ModAmp *= 1e-3 # T
    ModFreq *= 1e3 # Hz
    omegam = 2*np.pi*ModFreq #1/s

    M0 = 1
    gamma = constants.value(u'Bohr magneton')/constants.hbar*g

    # Estimator for maximum relevant Fourier order
    # (1) based on maximum field offset
    maxfieldoffset = np.max([ModAmp/2-DeltaB0,DeltaB0+ModAmp/2])
    maxfieldoffset = np.min([maxfieldoffset,ModAmp])
    maxfreqoffset = (gamma/2/np.pi)*maxfieldoffset
    kmax = maxfreqoffset/ModFreq
    kmax = np.ceil(kmax*1.4)

    # (2) based on T2
    # envelope = exp(-2/gam/ModAmp/T2*np.abs(fourierorder))
    threshold = 1e-6
    thresholdorder = -np.log(threshold)/2*gamma*ModAmp*T2
    thresholdorder = np.ceil(thresholdorder)

    # Combine (1) and (2)
    minkmax = 20
    kmax = np.max([kmax,thresholdorder])
    kmax = np.int(np.max([kmax,minkmax])) # at least 20

    # Solve Bloch equation for steady-state in frequency domain
    k = np.arange(-kmax-1,kmax+2,1,dtype=np.int32) # use max order kmax+1 to evaluate all terms
    a = 1j*k*omegam
    b = gamma*DeltaB0
    c = gamma*B1
    d = gamma*ModAmp/4
    tau1 = 1./(a+1/T1)
    tau2 = 1./(a+1/T2)

    q = np.arange(1,2*kmax+2,1,dtype=np.int32) # corresponds to -kmax:kmax range, changed indexing
    c2m = d**2*tau2[q-1]
    c2p = d**2*tau2[q+1]
    c1m = b*d*(tau2[q] + tau2[q-1])
    c1p = b*d*(tau2[q] + tau2[q+1])
    c0 = a[q] + 1/T2 + b**2*tau2[q] + c**2*tau1[q] + d**2*(tau2[q-1] + tau2[q+1])
    cL = c*M0*tau1[kmax+1]/T1 # changed index

    # Assemble pentadiagonal coefficient matrix and RHS vector and solve sparse
    # linear system of equations P*Y = C0 to get the Fourier coefficients Y = Mky
    nRows = np.int(2*kmax+1);
    C0 = lil_matrix((nRows,1),dtype=np.complex128) # use sparse lil_matrix for C0
    #C0 = csr_matrix((cL,(kmax,1)),shape=(nRows,1),dtype=np.complex128)

    C0[kmax] = cL; # changed to pythonic indexing
    C0 = C0.tocsr() # convert lil_matrix to csr matrix due to better linalg performance
    # Construct P as a sparse matrix from the diagonals
    P = diags([c2m, c1m, c0, c1p, c2p],[0, 1, 2, 3, 4],shape=(nRows,nRows+4),format='csr')

    P = P[:,2:nRows+2] # extract the correct columns to solve equation
    Y = spsolve(P,C0) # the result is already full, hopefully

    # Calculate Mkx and Mkz from Mky

    q = np.arange(1,2*kmax) # drop one order (max order now is kmax-1)
    #if ~onlyAbsorption
    Xk = tau2[q+1]*(b*Y[q] + d*(Y[q-1] + Y[q+1]))
    deltak0 = np.zeros((2*kmax-1,1))
    deltak0[kmax-1] = 1
    Zk = tau1[q+1]*(-c*Y[q] + M0*deltak0.transpose()/T1)
    Zk = Zk[0]
    #end
    Yk = Y[q];

    # Sparse-to-full conversion (since ifft does not support sparse) is not necessary since Y is already full
    #Yk = Yk.A;
    #if ~onlyAbsorption
    #Xk = Xk.A;
    #Zk = Zk.A;
    #end

    # Check if nPoints is given as input argument. Automatically set nPoints if missing.
    if nPoints==0:
        nPoints = 2*kmax-1;

    tPeriod = 1/ModFreq # modulation period
    t = np.linspace(0,tPeriod,nPoints,endpoint=False)

    # Inverse fourier transform to obtain transient EPR signal
    n = len(Yk)

    if nPoints<=n:
        # Fourier transform
        My = n*np.real(np.fft.ifft(np.fft.ifftshift(Yk))) # maybe the Python analogue of ifft(Y,'symmetric') is np.fft.ihfft(Y)
        #if ~onlyAbsorption
        Mx = n*np.real(np.fft.ifft(np.fft.ifftshift(Xk)))
        Mz = n*np.real(np.fft.ifft(np.fft.ifftshift(Zk)))
        #end
        if nPoints<n:
            tt = np.linspace(0,tPeriod,n,endpoint=False)
            My = np.interp(t,tt,My)
            #if ~onlyAbsorption
            Mx = np.interp(t,tt,Mx)
            Mz = np.interp(t,tt,Mz)
            #end
            #end
    else:
        fill = lambda m: np.concatenate((m[0:kmax], np.zeros(nPoints-2*kmax+1), m[kmax:len(m)]))
        FT = lambda m: nPoints*np.real(np.fft.ifft(fill(np.fft.ifftshift(m))))
        My = FT(Yk)
        #if ~onlyAbsorption
        Mx = FT(Xk)
        Mz = FT(Zk)
        #end

    return t,Mx,My,Mz

if __name__ == '__main__':
    main()
