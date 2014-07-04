import numpy as np
import matplotlib.pyplot as pl
import scipy.special as sp
from matplotlib.ticker import MaxNLocator
from scipy.integrate import simps
import scipy.signal as sg
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


def plotResults(fileHyperPar, directory, outFile):
	samples = np.load(fileHyperPar)
	ch = samples.reshape((300*1000,10)).T
	ch = ch[:,np.random.permutation(ch.shape[1])]

	files = glob.glob(directory+'*.dat')
	files.sort()

	nFiles = len(files)
	length = 500
					
	Sigma = np.zeros((nFiles,length))
	Y = np.zeros((nFiles,length))
	N = np.zeros((nFiles,length))
	Q = np.zeros((nFiles,length))
	Tau = np.zeros((nFiles,length))
									
	for i, f in enumerate(files):
		order = np.arange(length)
		np.random.shuffle(order)
		dat = np.loadtxt(f)
		print i, f
		Sigma[i,:] = dat[order[0:length],0]
		Y[i,:] = dat[order[0:length],1]
		N[i,:] = dat[order[0:length],2]
		Q[i,:] = dat[order[0:length],3]
		Tau[i,:] = dat[order[0:length],4]
		
	samples = [Sigma,Y,N,Q,Tau]
	pl.close('all')

	#fig, ax = pl.subplots(nrows=10, ncols=2, figsize=(10,12))

	loop = 1
	nTicks = 5
	labels = [r'$\alpha_\sigma$',r'$\beta_\sigma$',r'$\alpha_Y$',r'$\beta_Y$',r'$\alpha_N$',r'$\beta_N$',r'$\alpha_q$',r'$\beta_q$',r'$\alpha_\tau$',r'$\beta_\tau$']

	#for i in range(10):
		#mn = np.mean(ch[i,:])
		#st = np.std(ch[i,:])
		#ax[i,0].plot(ch[i,:], color='#969696')
		#ax[i,0].set_xlabel('Iteration')
		#ax[i,0].set_ylabel(labels[i])
		#ax[i,0].xaxis.set_major_locator(MaxNLocator(nTicks))
		#ax[i,0].set_ylim((0,10*st))
		#ax[i,1].hist(ch[i,:], color='#507FED', normed=True, bins=100)	
		#ax[i,1].set_xlabel(labels[i])
		#ax[i,1].set_ylabel('p('+labels[i]+'|D)')
		#ax[i,1].xaxis.set_major_locator(MaxNLocator(nTicks))	
		#ax[i,1].set_xlim((0,10*st))
		

	lower = [15.0, 5.0, 1.0, 0.0, 5.0]
	upper = [70.0, 30.0, 15.0, 3.0, 150.0]

	upperY = [0.3,0.16,4.0,4.0,0.1]
	pars = [r'$\sigma$','Y','N','q',r'$\tau$']

	fig, ax = pl.subplots(nrows=2, ncols=5, figsize=(17,10))
	for i in range(5):
		left = lower[i]
		right = upper[i]
		x = np.linspace(left+1e-2,right-1e-2,100)
		for j in range(100):
			alpha = ch[2*i,j]
			beta = ch[2*i+1,j]
			pf = betaPrior(x, alpha, beta, left, right)
			ax[0,i].plot(x, pf, alpha=0.3, color='#969696')		
		alpha = ch[2*i,0:1000]
		beta = ch[2*i+1,0:1000]
		p = betaAvgPrior(x, alpha, beta, left, right)
		ax[0,i].plot(x, p, color='#507FED', linewidth=2)
		ax[0,i].set_xlim((lower[i],upper[i]))
		ax[0,i].set_ylim((0,upperY[i]))	
		
		for j in range(nFiles):
			quantiles = stat.mstats.mquantiles(samples[i], prob=[0.5-0.68/2, 0.5, 0.5+0.68/2], axis=1)
			ax[1,i].plot(quantiles[j,1], j+1, 'o', color='#507FED')
			ax[1,i].errorbar(quantiles[j,1], j+1, xerr=[[quantiles[j,1]-quantiles[j,0]],[quantiles[j,2]-quantiles[j,1]]], color='#969696')
		ax[1,i].set_xlim((lower[i],upper[i]))
		ax[1,i].set_ylim((0,nFiles+2))
		ax[1,i].set_xlabel(pars[i])
		
	pl.tight_layout()

	fig.savefig(outFile)
		
		
	#mu = np.linspace(left + 1e-2,right - 1e-2,100)
	#pmu = np.zeros(100)
	#alpha = ch[2,:]
	#beta = ch[3,:]
	#pmu = betaAvgPrior(mu, alpha, beta, left, right)
	#ax = fig1.add_subplot(2,5,10)
	#ax.plot(mu,pmu, color='#507FED', linewidth=2)
	##pBTypeII = betaAvgPrior(mu, np.mean(ch[2,:]), np.mean(ch[3,:]), left, right)
	##ax.plot(mu,pBTypeII, '--', color='#969696', linewidth=2)
	#ax.set_xlabel(r'$\mu$')
	#ax.set_ylabel(r'$\langle$ p($\mu$|D) $\rangle$')
	#ax.xaxis.set_major_locator(MaxNLocator(nTicks))

plotResults('samplesHyperParType1.npy', 'onlyLinear/Type-1/', 'finalType1.pdf')
plotResults('samplesHyperParType2.npy', 'onlyLinear/Type-2/', 'finalType2.pdf')
plotResults('samplesHyperParNHBLR.npy', 'onlyLinear/NHBLR/', 'finalNHBLR.pdf')
