import numpy as np
import matplotlib.pyplot as pl
import scipy.special as sp
import scipy.misc as mi
from scipy.optimize import fsolve
import glob
from DIRECT import solve
import emcee
import pdb

class impSampling(object):
	def __init__(self, lower, upper, length):
		self.files = glob.glob('onlyLinear/*.dat')
		self.files.sort()
		
		self.nFiles = len(self.files)
		self.length = length
		
		self.Sigma = np.zeros((self.nFiles,self.length))
		self.Y = np.zeros((self.nFiles,self.length))
		self.N = np.zeros((self.nFiles,self.length))
		self.Q = np.zeros((self.nFiles,self.length))
		self.Tau = np.zeros((self.nFiles,self.length))
								
		for i, f in enumerate(self.files):
			order = np.arange(self.length)
			np.random.shuffle(order)
			dat = np.loadtxt(f)
			print i, f
			self.Sigma[i,:] = dat[order[0:self.length],0]
			self.Y[i,:] = dat[order[0:self.length],1]
			self.N[i,:] = dat[order[0:self.length],2]
			self.Q[i,:] = dat[order[0:self.length],3]
			self.Tau[i,:] = dat[order[0:self.length],4]
		
		self.lower = np.asarray(lower)
		self.upper = np.asarray(upper)						
		
	def logLike(self, x):
		alphaSigma, betaSigma, alphaY, betaY, alphaN, betaN, alphaQ, betaQ, alphaTau, betaTau = x
		nu = 0.001
				
# Priors
		priorSigma = - 2.5 * np.log(alphaSigma + betaSigma)
		priorY = - 2.5 * np.log(alphaY + betaY)
		priorN = - 2.5 * np.log(alphaN + betaN)
		priorQ = - 2.5 * np.log(alphaQ + betaQ)
		priorTau = - 2.5 * np.log(alphaTau+ betaTau)
		
		lnP = priorSigma + priorY + priorN + priorQ + priorTau
		
		cteSigma = (1.0-alphaSigma-betaSigma) * np.log(self.upper[0]-self.lower[0]) - sp.betaln(alphaSigma,betaSigma)
		cteY = (1.0-alphaY-betaY) * np.log(self.upper[1]-self.lower[1]) - sp.betaln(alphaY,betaY)
		cteN = (1.0-alphaN-betaN) * np.log(self.upper[2]-self.lower[2]) - sp.betaln(alphaN,betaN)
		cteQ = (1.0-alphaQ-betaQ) * np.log(self.upper[3]-self.lower[3]) - sp.betaln(alphaQ,betaQ)
		cteTau = (1.0-alphaTau-betaTau) * np.log(self.upper[4]-self.lower[4]) - sp.betaln(alphaTau,betaTau)
						
		vecSigma = cteSigma + (alphaSigma-1.0) * np.log(self.Sigma-self.lower[0]) + (betaSigma-1.0) * np.log(self.upper[0]-self.Sigma)
		vecY = cteY + (alphaY-1.0) * np.log(self.Y-self.lower[1]) + (betaY-1.0) * np.log(self.upper[1]-self.Y)
		vecN = cteN + (alphaN-1.0) * np.log(self.N-self.lower[2]) + (betaN-1.0) * np.log(self.upper[2]-self.N)
		vecQ = cteQ + (alphaQ-1.0) * np.log(self.Q-self.lower[3]) + (betaQ-1.0) * np.log(self.upper[3]-self.Q)
		vecTau = cteTau + (alphaTau-1.0) * np.log(self.Tau-self.lower[4]) + (betaTau-1.0) * np.log(self.upper[4]-self.Tau)
								
		lnP += np.sum(mi.logsumexp(vecSigma + vecY + vecN + vecQ + vecTau - np.log(self.length), axis=1))
		#print lnP
		
		return lnP
	
	def equations(self, p, a, b, meanX, varX):
		alpha, beta = p
		return ( (alpha*b+beta*a) / (alpha+beta) - meanX, (alpha*beta*(b-a)**2) / ((alpha+beta)**2 * (alpha+beta+1.0)) - varX )
	
	def betaDistribution(self, x, alpha, beta, a, b):
		return (x-a)**(alpha-1.0) * (b-x)**(beta-1.0)		
	
	def initialValues(self):
		pl.close('all')
		fig, ax = pl.subplots(nrows = 5, ncols=2, figsize=(12,10))
		dat = [self.Sigma, self.Y, self.N, self.Q, self.Tau]
		bestFit = np.zeros((5,2))
		for i in range(5):
			alpha, beta = fsolve(self.equations, (1, 1), args=(self.lower[i], self.upper[i], np.mean(dat[i]), np.var(dat[i])))
			bestFit[i,:] = [alpha, beta]
			print alpha, beta
			ax[i,0].hist(dat[i].flatten())
			xAxis = np.linspace(self.lower[i], self.upper[i], 100)			
			ax[i,1].plot(xAxis, (xAxis-self.lower[i])**(alpha-1.0) * (self.upper[i]-xAxis)**(beta-1.0))
		pl.tight_layout()
		return bestFit.flatten()		
				
	def logPrior(self, x):		
		if (np.all(x > 0)):
			return 0.0
		return -np.inf
		
	def logPosterior(self, x):
		logP = self.logPrior(x)
		if (not np.isfinite(logP)):
			return -np.inf
		return logP + self.logLike(x)
			
	def sample(self):
		ndim, nwalkers = 10, 300
		p0 = np.zeros((nwalkers,ndim))
		initial = self.initialValues()
		p0 = emcee.utils.sample_ball(initial, 0.1*np.ones(ndim), size=nwalkers)
					
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logPosterior)
		self.sampler.run_mcmc(p0, 5000)
		
lower = [15.0, 5.0, 1.0, 0.0, 5.0]
upper = [70.0, 100.0, 15.0, 3.0, 150.0]

out = impSampling(lower, upper, 200)
out.sample()

samples = out.sampler.chain[:,-1000:,:]

np.save('samplesHyperPar.npy',samples)

