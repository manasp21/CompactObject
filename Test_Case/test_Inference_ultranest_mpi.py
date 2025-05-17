#Package we need:
import InferenceWorkflow.BayesianSampler as sampler
import InferenceWorkflow.Likelihood as likelihood
import InferenceWorkflow.prior as prior
import math
import numpy as np

from TOVsolver.constant import oneoverfm_MeV, m_rho, m_w,G,c
import TOVsolver.main as main
import EOSgenerators.crust_EOS as crust
import EOSgenerators.fastRMF_EoS as RMF
import warnings


#Tolos_crust_out = np.loadtxt('Tolos_crust_out.txt', delimiter='  ')
Tolos_crust_out = np.loadtxt('Tolos_crust_out.txt', delimiter=None, comments='#', usecols=(0, 1, 2, 3, 4))
eps_crust_out = Tolos_crust_out[:,3] * G / c**2
pres_crust_out = Tolos_crust_out[:,4] * G / c**4

eps_crust, pres_crust = crust.PolyInterpolate(eps_crust_out, pres_crust_out)

parameters = ['g_sigma', 'g_omega','g_rho', 'kappa', 'lambda_0', 'zeta', 'Lambda_w','d1']

def prior_transform(cube):
    params = cube.copy()
    params[0] = math.sqrt(prior.normal_Prior(107.5, 7.5,cube[0]))
    params[2] = math.sqrt(prior.flat_prior(75,210,cube[2]))
    params[1] = math.sqrt(prior.flat_prior(150,210,cube[1]))
    params[3] = prior.normal_Prior(2.525/oneoverfm_MeV, 1.525/oneoverfm_MeV,cube[3])
    params[4] = prior.normal_Prior(0.0045, 0.0205,cube[4])
    params[5] = prior.flat_prior(0,0.04,cube[5])
    params[6] = prior.flat_prior(0,0.045,cube[6])
    
    g_sigma = params[0]
    g_omega = params[1]
    g_rho = params[2]

    kappa = params[3]
    lambda_0 = params[4]
    zeta = params[5]
    Lambda_w = params[6]
    m_sig = 495 / oneoverfm_MeV

    theta = np.array([m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa,
                         lambda_0, zeta, Lambda_w])
    
    ep, pr = RMF.compute_EOS(eps_crust, pres_crust, theta)

    eps_total = np.hstack((eps_crust,ep))
    
    pres_total = np.hstack((pres_crust,pr))
    
    RFSU2R = []
    MFSU2R = []
    density = np.logspace(14.3, 15.6, 50)
    if all(x<y for x,y in zip(eps_total[:], eps_total[1:])) and all(x<y for x, y in zip(pres_total[:], pres_total[1:])):
        MR = main.OutputMR('',eps_total,pres_total).T
        if len(MR[1]) == False: 
            params[7] = 0
            #params[8] = 0
            # this line for showing how to add one more observation
        else:
   
            for i in range(len(MR[1])):
                RFSU2R.append(MR[0][i])
                MFSU2R.append(MR[1][i])   
                if i > 20 and MR[1][i] - MR[1][i-1]< 0:
                    break
    if len(MFSU2R)==False:
        params[7] = 0
        # params[8] = 0
        # this line for showing how to add one more observation
    else:
        max_index = len(MFSU2R)
        max_d = np.log10(density[max_index-1])
        params[7] = 14.3 + (max_d - 14.3) * cube[7]
        # params[8] = 14.3 + (max_d - 14.3) * cube[8]
        # this line for showing how to add one more observation
    return params


import scipy.stats as stats

import scipy.stats as stats
def likelihood_transform(theta):
    # This is a demonstration code for only introduce one constraint from one mass-radius observation.
    # Could be very easy to implement more constraint from nuclear quantity, since that do not need to
    # sample more central density of real neutron star. If user want to expand to two mass radius measurement 
    # the code could be:
    
    # g_sigma, g_omega,g_rho, kappa, lambda_0, zeta, Lambda_w, d1, d2 = theta
    g_sigma, g_omega,g_rho, kappa, lambda_0, zeta, Lambda_w, d1 = theta # comment this line if you need two measuremnts.
    
    ####################################################################################################################
    ############ This is the block to compute out all the EoS you need based on your parameters#########################
    m_sig = 495 / oneoverfm_MeV
    m_w = 3.96544
    m_rho = 3.86662
    theta1 = np.array([m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w])
    ep, pr = RMF.compute_EOS(eps_crust, pres_crust, theta1)

    eps_total = np.hstack((eps_crust,ep))
    pres_total = np.hstack((pres_crust,pr))
    ####################################################################################################################
    
    # probMRgaussian1 = likelihood.MRlikihood_Gaussian(eps_total,pres_total,(1.4,13,0.07,0.65),d1)
    # probMRgaussian2 = likelihood.MRlikihood_Gaussian(eps_total,pres_total,(2.0,12,0.1,0.6),d2)
    # probMR = probMRgaussian1 + probMRgaussian2
    
    # Same could be extended to more distributions. Notice the prior definition should be change accordingly
    # by define more density parameters like here d2.
    
    #1. This line is to compute MR likelihood from a Simulated MR measurement:
    
    
    probMRgaussian = likelihood.MRlikihood_Gaussian(eps_total,pres_total,(1.4,13,0.07,0.65),d1)
    
    #2. This is  a block that constrain from given real MR measurement, say J0030:
    #J0030 = numpy.loadtxt('data/PST_equal_sampled_MR.txt', delimiter=' ')
    #J30R_list, J30M_list = zip(*J0030)
    #J30R_list = numpy.array(J30R_list).T    
    #J30M_list = numpy.array(J30M_list).T
    #Rmin = J30R_list.min()
    #Rmax = J30R_list.max()
    #Mmin = J30M_list.min()
    #Mmax = J30M_list.max()
    #X3, Y3 = numpy.mgrid[Rmin:Rmax:500j, Mmin:Mmax:100j]
    #positions = numpy.vstack([X3.ravel(), Y3.ravel()])
    #values = numpy.vstack([J30R_list, J30M_list])
    #kernel3 = stats.gaussian_kde(values)
    #probMRJ0030 = likelihood.MRlikelihhood_kernel(eps_total,pres_total,kernel3,d1)
    
    #3. This is to compute the constraint from experiment of nuclearmatter
    # 250<K<400, 25<J<38, 30<L<86:
    # hint: since this K,J,L sampling don't need to sample central density so this 
    # theta should be redifined.
    #probK = likelihood.Kliklihood(theta,250,400)
    #probJ = likelihood.Jliklihood(theta,250,400)
    #probL = likelihood.Lliklihood(theta,250,400)
    
    #4. This is block to cosntrain from a GW event, say GW170817, here the file of this
    # event is origanized by [chrip mass, M2/M1, tidal of M1, tidal of M2, sampling weight]:
    #GW170817 = np.load('GW170817_McQL1L2weights.npy')
    #chrip170817 = stats.gaussian_kde(GW170817[:,0],weights = GW170817[:,4])
    #kernelGW = stats.gaussian_kde(GW170817.T[0:4],weights = GW170817[:,4])
    #probGW = likelihood.TidalLikihood_kernel(eps_total,pres_total,(kernelGW,chrip170817),d1)


    #5. This block constraints pure neutron matter EOS
    ## For pure neutron matter alpha = 0
    #theta1    = np.append(theta, 0)
    #rho, e, p = RMF.get_eos_alpha(theta1)
    #EoS_PNM   = np.array([rho, e*oneoverfm_MeV, p*oneoverfm_MeV]) # energy density and pressure are converted from fm^-4 to MeV fm^-3
 
    #probPNM = likelihood.chiEFT_PNM(EoS_PNM, type="Super Gaussian", contraint_quantity="e", enlargement=0.1) # 10% enlargement 

    prob =  probMRgaussian #+ probMRJ0030 + probK + probJ + probL + probGW + probPNM 
    return prob
    
    
step = 2 * len(parameters)
live_point = 400     # For production runs, increase up to 2000 for optimal results

max_calls = 10000    # For production runs, set this as high as computationally feasible
samples = sampler.UltranestSampler(parameters,likelihood_transform,prior_transform,step,live_point,max_calls)
