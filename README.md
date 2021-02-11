#Curve-fitting-HW1-Math466
#HW1- Data fitting with MLE
# 0. Import the necessary libraries
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt # minimizing procedure
from scipy.interpolate import*
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#to plot within notebook
import matplotlib.pyplot as plt
# Fitting Polynomial Regression to the dataset 

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
# Display settings
# read csv data
df = pd.read_csv('C:/Users/saeid/OneDrive/Documents/claremont/466/Projrct 1/TSLA.csv')
print (df.columns)

#to plot within notebook
import matplotlib.pyplot as plt
y = np.asarray(df['High'])
x = np.asarray(df.index.values)
plt.plot(x,y,label='Tesla''b-')
plt.xlabel('Days')
plt.ylabel('Tesla value')
plt.title('Data Fitting (polynomial and MLE)')
plt.legend()
#linear regression
p1=np.polyfit(x,y,1)
print(p1)
plt.plot(x,np.polyval(p1,x),'r--',label='1st')
#Choose the order of your polynomial. Here the degree is set to 5.
#hence the mathematical model equation is 
#y = c0 + c1.x**1 + c2.x**2+....+ c5.x**5
p2=np.polyfit(x,y,2)
print(p2)
plt.plot(x,np.polyval(p2,x),'g--',label='2nd')
#Polynamial 3rd degree
p3=np.polyfit(x,y,3)
print(p3)
plt.plot(x,np.polyval(p3,x),'m--',label='3rd')
#Polynamial 4th degree
p4=np.polyfit(x,y,4)
print(p4)
plt.plot(x,np.polyval(p4,x),'y--',label='4th')
#Polynamial 100th degree
p100=np.polyfit(x,y,100)
print(p100)
plt.plot(x,np.polyval(p100,x),'c',label='100')

 #MLE
 
Data = df['High']                                     # Load data
Data = Data.dropna()
Data = np.log(Data)                                   # Take log() transformation

# Check for right answers estimate
print('mean Data (check)= ', Data.mean(), 'std dev. Data (check) = ', Data.std() )
# ----------------------------------------------------------------------
# 2. Define function that generates probabilities from a normal pdf
# ----------------------------------------------------------------------
def norm_pdf(xvals, mu, sigma):
    pdf_vals = ((1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (xvals - mu) ** 2 / (2 * sigma ** 2)))) # normal pdf distribution
    return pdf_vals
# ----------------------------------------------------------------------
# 3. Define function that provides the log likelihood
# ----------------------------------------------------------------------
def log_lik_norm(xvals, mu, sigma):
    # Generates values from a normal distributed pdf
    pdf_vals = norm_pdf(xvals, mu, sigma)
    # Take log of normal distributed pdf values
    ln_pdf_vals = np.log(pdf_vals)
    # Summation of normal distributed pdf values (= loglikelihood)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val
# ----------------------------------------------------------------------
# 4. Define criterion function
# ----------------------------------------------------------------------
def crit(params, *args): # Provides the negative log-likelihood
    # Parameters
    mu, sigma = params
    # Arguments
    xvals = args
    # Log-likelihood
    log_lik_val = log_lik_norm(xvals, mu, sigma)
    # Maximizing -> NOTE: Minimizing is computationally more stable
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val
# ----------------------------------------------------------------------
# 6. Maximization procedure of criterion function
# ----------------------------------------------------------------------
# Initialization of parameters
mu_init = 3
sig_init = 1 # initalization is sensitive for 'L-BFGS-B' -> first guess: method='SLSQP'
params_init = np.array([mu_init, sig_init])
# Arguments
mle_args = (Data)
# Minimizing procedure (constrained)
results_cstr = opt.minimize(crit, params_init, args=(mle_args), method='SLSQP',
                            bounds=((None, None), (1e-10, None)))
print(results_cstr)
# Maximum Likelihood Estimators
mu_MLE, sig_MLE = results_cstr.x
print('mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)
# ----------------------------------------------------------------------
# 7. Plot the results of MLE
# ----------------------------------------------------------------------
# Plot the MLE estimated distribution
dist_Data = np.linspace(0, 10, 500) # generates chunk of data for pdf
plt.plot(dist_Data, norm_pdf(dist_Data, mu_MLE, sig_MLE),
         linewidth=2, color='k', label='3: $\mu$={},$\sigma$={}'.format(mu_MLE,sig_MLE))
plt.legend(loc='upper left')
#
plt.show()

# Print log-likelihoods for comparison
print('MLE log-likelihood 3: ', log_lik_norm(Data, mu_MLE, sig_MLE)) #MLE (clearly maxmized)

print('MLE log-likelihood check: ', log_lik_norm(Data, Data.mean(), Data.std())) #MLE (clearly maxmized)
# ----------------------------------------------------------------------
# 8. Plot log likelihood around M.L. Estimates for mu and sigma
# ----------------------------------------------------------------------
cmap1 = matplotlib.cm.get_cmap('summer')
#
mu_vals = np.linspace(0.75*mu_MLE, 1.25*mu_MLE, 50) # values around mu_MLE
sig_vals = np.linspace(0.75*sig_MLE, 1.25*sig_MLE, 50) # values around sig_MLE
lnlik_vals = np.zeros((50, 50))
for mu_ind in range(50):
    for sig_ind in range(50):
        lnlik_vals[mu_ind, sig_ind] = \
            log_lik_norm(Data, mu_vals[mu_ind],
                              sig_vals[sig_ind])

mu_mesh, sig_mesh = np.meshgrid(mu_vals, sig_vals)
#
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(sig_mesh, mu_mesh, lnlik_vals, rstride=8,
                cstride=1, cmap=cmap1)
ax.set_title('Log likelihood for values of mu and sigma')
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\mu$')
ax.set_zlabel(r'log likelihood')
plt.show()
