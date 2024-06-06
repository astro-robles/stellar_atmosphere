from astropy import units as u
from astropy import constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import special
from scipy.optimize import curve_fit
from astropy.modeling.models import BlackBody
import pandas as pd

a0 = 1.0443*10**(-26) # with wavelength in angstroms
R = 1.09678*10**(-3) #per angstrom
Rcm = 1.09678*10**(5) #per cm
loge = .43429
h = const.h.cgs.value
e = const.e.gauss.value
me = const.m_e.value
pi = np.pi
c = const.c.cgs.value
ae = const.sigma_T.cgs.value
Na = const.N_A.value

def load_stuff():
    load_partition_data()
    load_ionization_data()
    load_abundance_data()


## Planck function with frequency
# Frequency in Hz
# Temperature in K
def Planck(freq,temp):
    num = 2 * const.h * (freq)**3
    den = (const.c**2) * (np.exp(const.h*(freq)/(const.k_B * temp))-1) * u.sr
    intensities = num/den
    return intensities

## Planck function with wavenumber
# Wavenumber needs to be 1/length units
# Temperature in K
def Planck_num(wavenumbers,temp):
    converted_to_freq = (wavenumbers).to(u.Hz, equivalencies=u.spectral())
    num = 2 * const.h * (converted_to_freq)**3
    den = (const.c**2) * (np.exp(const.h*(converted_to_freq)/(const.k_B * temp))-1) * u.sr
    intensities = num/den
    return intensities


## Simple Box Integrator
# Also works if x and y have astropy units
def Box_Integrator(xvalues,yvalues):
    sum = 0
    for x in range(xvalues.size-1):
        deltax = xvalues[x+1] - xvalues[x]
        area = deltax * yvalues[x]
        sum += np.nan_to_num(area)       
    return sum

def diff(yvalues):
    diff_values = np.zeros(np.size(yvalues)-1)
    for x in range(yvalues.size-1):
        diff_values[x] = np.absolute(yvalues[x+1] - yvalues[x])
  
    return diff_values

def Precision(truth,calc):
    return np.absolute(1-(calc/truth))

def load_partition_data():
    global partition_species
    global partition_coef
    global partition_data
    
    df    = pd.read_csv('RepairedPartitionFunctions.txt',header=None,sep=" ")
    df    = df.replace('-',np.nan)
    tempy = df.to_numpy()
    b     = tempy[:,1:-1]
    s     = tempy[:,0]

    partition_data    = b.astype(float)
    partition_species = s.astype(str)

    ## interpolating
    theta_columns       = np.linspace(.2,2.0,num=10)  
    # to match with the input data
    num_rows            = np.shape(b)[0]
    partition_coef      = np.zeros([num_rows,3])
    
    def test_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    for i in range(num_rows):
        idx               = np.isfinite(theta_columns) & np.isfinite(partition_data[i])
        #partition_coef[i] = np.polyfit(theta_columns[idx],partition_data[i][idx],7)
        try:
            partition_coef[i], param_cov = curve_fit(test_func, theta_columns[idx], partition_data[i][idx])
        except:
            pass
def partition(temp,specy):
    # load partition info
    if specy == 'H-' or specy == 'HII' or specy == 'Li+' or specy =='Cs+':
        if isinstance(temp, int) or isinstance(temp, float):
            return 1
        else:
            return np.ones(temp.size)
    else:
        # find the species
        spec_index = np.where(partition_species == specy)[0][0]
    
        # interpolate function
        #spec_function = np.poly1d(partition_coef[spec_index])
        theta         = temptotheta(temp)
        param = partition_coef[spec_index]
        ans = param[0] * np.exp(-param[1] * theta) + param[2]
        return np.power(10,ans) 
        
def Phi(temp,spec):
    # temperature dependent part of RHS Saha
    # find next ionization
    if spec == 'H-':
        nextspec = 'H'
    elif spec == 'H':
        nextspec = 'HII'
    else:
        nextspec = spec + '+'
    u0    = partition(temp,spec)
    u1    = partition(temp,nextspec)
    theta = temptotheta(temp)
    I     = X(spec)
    return (1.2024*10**9)*(u1/u0)*np.power(theta,-2.5)*np.power(10,-theta*I)

def X(specy):
    if specy == 'H-':
        return 0.755
    else:
        # check nist
        nist_index = np.where(nist_data == specy)[0][0]
        if nist_index>0:
            #use nist
            return nist_data[nist_index,2]
        else:
            #use grey        
            #how many +s
            numplus= len(specy)-specy.find('+')
            if specy.find('+')>0:
                specy = specy[:-numplus]
            else:
                numplus=0
    
            spec_index = np.where(ionization_data == specy)[0][0]
            
            return ionization_data[spec_index,numplus+2]

def load_ionization_data():
    global ionization_data
    global nist_data
    
    df2             = pd.read_fwf('ioniz.txt',header=None)  
    iontemp         = df2.to_numpy()
    ionization_data = iontemp[:,1:]

    df3 = pd.read_csv('nist_ioniz2.csv',header=None,sep = ',')
    nisttemp = df3.to_numpy()
    nist_data = nisttemp[:,1:]
    testing = nist_data[:,0]
    nist_data[:,0] = np.char.strip(testing.astype(str))

def thetatotemp(theta):
    return 5040./theta

def temptotheta(temp):
    return 5040./temp

## hw7 OPACITY stuff #############################################

def alpha_H_bf(wl,n): # wl is wavelength
    return a0*g_bf(wl,n)*(wl**3)*(n**(-5))
    # returns units of cm2/nuetral hydrogen absorber
def g_bf(wl,n):
    first  = np.ones(wl.size)
    second = .3456/((wl*R)**(1/3))
    third  = ((wl*R)/(n**2))-(.5)
    return first-(second*third)

def k_H_bf(wl,temp,Pe):
    ones = np.ones(wl.size)
    n0 = np.floor(np.sqrt(wl*R)+ones)
    #print("n0 is"+str(n0))
    I = X('H')
    b3 = I*(ones-(1/(n0+3)**2))
    theta = temptotheta(temp)

    Sigma = 0
    for x in range(3):
        n = n0 + x
        Sigma += (g_bf(wl,n)/(n**3))*np.power(10,-theta*H_X(n))
    
    second = (loge/(2*theta*I))*(np.power(10,-b3*theta)-np.power(10,-I*theta))
    
    return a0*(wl**3)*(Sigma+second)
    
def H_X(n):
    return 13.598*(np.ones(n.size)-(1/(n**2)))
    
def alpha_H_ff(wl,temp):
    wlcm=wl*10**(-8)
    num = 2*h**2*e**2*Rcm*wlcm**3*(2*me)**(1/2)
    den = 3**(3/2)*pi*me**3*c**3*(pi*k*temp)**(1/2)
    return num/den
    # in cm^2 per absorber per electron?

def g_ff(wl,temp):
    theta = temptotheta(temp)
    ones = np.ones(wl.size)
    chi_wl = (1.2398*10**4)/wl
    return ones+((.3456/(wl*R)**(1/3))*((loge/(theta*chi_wl))+.5))

def k_H_ff(wl,temp,Pe):
    I = X('H')
    theta = temptotheta(temp)
    ex=-theta*I
    return a0*wl**3*g_ff(wl,temp)*(loge/(2*theta*I))*np.power(10,ex)

def alpha_H_neg_bf(wl):
    a = np.zeros(7)
    a[0]=.1199654
    a[1]=-1.18267*10**(-6)
    a[2]=2.64243*10**(-7)
    a[3]=-4.40524*10**(-11)
    a[4]=3.23992*10**(-15)
    a[5]=-1.39568*10**(-19)
    a[6]=2.78701*10**(-24)
    print(a)
    # sum=np.zeros(wl.size)
    # for x in range(7):
    #     sum+=a[x]*np.power(wl,x)
    final = np.zeros(wl.size)
    for x in range(wl.size):
        if wl[x]<16110:
            final[x]=a[0] + a[1]*wl[x] + a[2]*wl[x]**2 + a[3]*wl[x]**3 + a[4]*wl[x]**4 + a[5]*wl[x]**5 + a[6]*wl[x]**6
            
    return final*10.


def k_H_neg_bf(wl,temp,Pe):
    theta = temptotheta(temp)
    return (4.158*10**(-10))*10**(-18)*alpha_H_neg_bf(wl)*Pe*theta**(5./2.)*np.power(10,.754*theta)

def k_H_neg_ff(wl,temp,Pe):
    lwl=np.log10(wl)
    theta = temptotheta(temp)
    log_theta =  np.log10(theta)
    f0 = -2.2763-1.685*lwl+.76661*lwl**2-.0533464*lwl**3
    f1 = 15.2827-9.2846*lwl+1.99381*lwl**2-.142631*lwl**3
    f2 = -197.789+190.266*lwl-67.9775*lwl**2+10.6913*lwl**3-.625151*lwl**4
    return 10**(-26)*Pe*np.power(10,f0+f1*log_theta+f2*log_theta**2)

def k_e(Pe,Pg):
    return ae*(Pe)*np.nansum(abun_data[:,2])/(Pg-Pe)

def k_tot(wl,temp,Pe,Pg):
    ones = np.ones(Pe.size)
    kbf   = k_H_bf(wl,temp,Pe)
    kff   = k_H_ff(wl,temp,Pe)
    knbf  = k_H_neg_bf(wl,temp,Pe)
    knff  = k_H_neg_ff(wl,temp,Pe)

    Phi2    = Phi(temp,'H')
    chi_wl = (1.2398*10**4)/wl
    theta  = temptotheta(temp)
    es = k_e(Pe,Pg)
    #ktot = np.zeros([5,wl.size])
    #factor1  = (1-np.power(10,-chi_wl*theta))
    #factor2 = (1/(1+(Phi/Pe)))
    se_factor=(ones-np.power(10,-chi_wl*theta))
    print("simulated emission factor is:"+str(se_factor))
    h_neutral_frac=(ones/(ones+(Phi2/Pe)))
    print("H neutral fraction is:"+str(h_neutral_frac))
    sum = np.nansum(abun_data[:,2]*abun_data[:,1])/Na
    print("The total sum of abundances in g:"+str(sum))
    
    print("knbf no fluff"+str(knbf))
    print("knbf:"+str(knbf*se_factor*h_neutral_frac/sum))
    print("knff:"+str(knff*h_neutral_frac/sum))
    print("kbf:"+str(kbf*se_factor*h_neutral_frac/sum))
    print("kff:"+str(kff*se_factor*h_neutral_frac/sum))
    print("k_e:"+str(es/sum))
    return ((kbf+kff+knbf)*se_factor + knff)*h_neutral_frac+es

def k_nu(wl,temp,Pe,Pg):
    sum = np.nansum(abun_data[:,2]*abun_data[:,1])/Na # amu to g
    return k_tot(wl,temp,Pe,Pg)/sum

## from hw8 PRESSURE #############################################

def load_abundance_data():
    global abun_data
    global elements
    df   = pd.read_csv('SolarAbundance.txt',sep = '\t')  
    abun = df.to_numpy()
    abun_data = abun[:,1:]
    elements = abun_data[:,0]

def A(specy):
    # return the A for a given element
    spec_index = np.where(abun_data == specy)[0][0]
    return np.nan_to_num(abun_data[spec_index,2])

# def Pe(Pg,T):
#     pe  = Pe_guess(Pg,T)
#     tol=10**(-4)
#     dif=2
#     iter=0
#     while dif>tol and iter<100: 
#         iter+=1
#         #loop through j
#         sum1 = 0
#         sum2 = 0
#         for i in range(28):
#             Phi_j = sa.Phi(T,elements[i])
#             A_j   = A(elements[i])
#             #bigfrac = (Phi_j)/(1+frac)
#             sum1+=A_j*((Phi_j)/(1+(Phi_j/pe)))
#             sum2+=A_j*(1+((Phi_j/pe)/(1+(Phi_j/pe))))
#         result = np.sqrt(Pg*sum1/sum2)
#         dif    = np.abs(result-pe)
#         pe     = result
#     return result
def Pe(Pg,T):
    # needs numpy array inputs
    pe  = Pe_guess(Pg,T)
    tol=10**(-4)
    dif=2
    iter=0
    ones =np.ones(T.size)
    while np.any(dif>tol) and iter<100: 
        iter+=1
        #loop through j
        sum1 = np.zeros(T.size)
        sum2 = np.zeros(T.size)
        for i in range(28):
            Phi_j = Phi(T,elements[i])
            A_j   = A(elements[i])
            #bigfrac = (Phi_j)/(1+frac)
            sum1+=A_j*((Phi_j)/(ones+(Phi_j/pe)))
            sum2+=A_j*(ones+((Phi_j/pe)/(ones+(Phi_j/pe))))
        result = np.sqrt(Pg*sum1/sum2)
        dif    = np.abs(result-pe)
        pe     = result
    return result

def Pe_guess(Pg,T):
    # need to loop through T
    Tsize=T.size
    guess = np.zeros(Tsize)  
    for i in range(Tsize):   
        if T[i]>30000:
            guess[i] = Pg[i]/2
        else:
            guess[i] = np.sqrt(Pg[i]*Phi(T[i],'H'))
    return guess

