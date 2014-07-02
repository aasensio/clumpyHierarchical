import numpy as np
import matplotlib.pyplot as pl
import scipy.special as sp
from matplotlib.ticker import MaxNLocator
from scipy.integrate import simps
import scipy.stats as stat
import pdb
import glob

def betaAvgPrior(x, alpha, beta, left, right):
	Beta = sp.beta(alpha, beta)
	pf = np.zeros(len(x))
	for i in range(len(x)):
		ylog = ( (1.0-alpha-beta) * np.log(right-left) - (sp.gammaln(alpha) + sp.gammaln(beta) - sp.gammaln(alpha+beta)) +
			(alpha-1.0) * np.log(x[i] - left) + (beta-1.0) * np.log(right - x[i]) )		
		pf[i] = np.mean(np.exp(ylog))
	return pf

def betaPrior(x, alpha, beta, left, right):
	Beta = sp.beta(alpha, beta)
	pf = np.zeros(len(x))	
	ylog = ( (1.0-alpha-beta) * np.log(right-left) - (sp.gammaln(alpha) + sp.gammaln(beta) - sp.gammaln(alpha+beta)) +
		(alpha-1.0) * np.log(x - left) + (beta-1.0) * np.log(right - x) )
	pf = np.exp(ylog)
	return pf

def kullbackLeibler(x1, p1, x2, p2):
	minX = np.min((x1,x2))
	maxX = np.max((x1,x2))
	
	x = np.linspace(minX, maxX, 1000)
	
	p1Norm = np.interp(x, x1, p1)
	p2Norm = np.interp(x, x2, p2)
	
	norm1 = simps(p1, x=x1)
	norm2 = simps(p2, x=x2)
	
	p1Norm /= norm1
	p2Norm /= norm2
	
	p1Norm[p1Norm <= 0] = 1e-8
	p2Norm[p2Norm <= 0] = 1e-8
	
	return simps(p1Norm * (np.log2(p1Norm) - np.log2(p2Norm)), x=x)

def computeKLD():
	
	files = glob.glob('samplesHyperPar*.npy')
	files.sort()
			
	loop = 1
	nTicks = 5
	labels = [r'$\alpha_\sigma$',r'$\beta_\sigma$',r'$\alpha_Y$',r'$\beta_Y$',r'$\alpha_N$',r'$\beta_N$',r'$\alpha_q$',r'$\beta_q$',r'$\alpha_\tau$',r'$\beta_\tau$']

	lower = [15.0, 5.0, 1.0, 0.0, 5.0]
	upper = [70.0, 30.0, 15.0, 3.0, 150.0]

	upperY = [0.3,0.16,4.0,4.0,0.1]
	pars = ['sigma','Y','N','q','tau']
	
	names = ['NHBLR','Type1','Type2']
			
	n = len(files)
	xPrior = [[] for i in range(n)]
	pPrior = [[] for i in range(n)]		
	
	for k, f in enumerate(files):			
		samples = np.load(f)
		ch = samples.reshape((300*1000,10)).T
		ch = ch[:,np.random.permutation(ch.shape[1])]
		for i in range(5):
			left = lower[i]
			right = upper[i]
			x = np.linspace(left+1e-2,right-1e-2,100)
			for j in range(100):
				alpha = ch[2*i,j]
				beta = ch[2*i+1,j]
				pf = betaPrior(x, alpha, beta, left, right)
			alpha = ch[2*i,0:1000]
			beta = ch[2*i+1,0:1000]
			p = betaAvgPrior(x, alpha, beta, left, right)
			
			pPrior[k].append(p)
			xPrior[k].append(x)
			
	DKL = np.zeros((5,n,n))
	for i in range(5):
		for k1 in range(n):
			for k2 in range(n):
				DKL[i,k1,k2] = kullbackLeibler(xPrior[k1][i], pPrior[k1][i], xPrior[k2][i], pPrior[k2][i])
				print "{0} -> DKL({1},{2})={3}".format(pars[i],names[k1], names[k2], DKL[i,k1,k2])
				
computeKLD()