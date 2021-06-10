import numpy as np
import scipy.stats as st
import math
import copy

def mean(nlist):
	return np.mean(nlist)
def var(nlist):
	return np.var(nlist, ddof=1)

def genSample(k, n=1, RandomSeed=-1):
	mu = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	sigma = 2
	if RandomSeed>0:
		np.random.seed( RandomSeed )
	rand_data = np.random.normal(mu[k], sigma, n)
	return rand_data.tolist()


def KG(k, N, variance, pmiu, pvar): 
	# variance is the common and known variance parameter across all all alternatives
	# pmiu / pvar is a list in which the elements are the prior predictions of mean / variance of the mean parameter
	miuit = copy.deepcopy(pmiu)
	t = 0
	betait = 1 / np.array(pvar)
	beta = 1 / variance
	nt = np.zeros(k) # number of observations generated (each alternative)
	index = 0 # upper bound of index is 20  
	while t<N:
		if index > 20:
			break
		var_tilde = 1 / betait - 1 / ( betait + np.array([beta for i in range(0,k)]) )
		zeta = np.zeros(k)
		for i in range(0,k):
			miujt = max( [miuit[j] for j in range(0,k) if j!=i] )
			zeta[i] = -abs( (miuit[i] - miujt) / np.sqrt(var_tilde[i]) )
		Tlist = np.sqrt(var_tilde) * (zeta * st.norm.cdf(zeta) + st.norm.pdf(zeta))
		if sum(Tlist) == 0: # each element of Tlist is 0
			index += 1
		xt = np.argmax( Tlist ) # questions remain here
		yxt = genSample(k=xt, n=1) # note: yxt is a list with a single element
		nt[xt] += 1
		betait[xt] = betait[xt] + beta # update betait[xt]
		miuit[xt] = ((betait[xt]-beta) * miuit[xt] + beta * yxt[0]) / betait[xt]
		t += 1
	maxi = np.argmax(miuit)
	return {'index': maxi, 'N': np.round(nt), 'means': miuit, 'total': t} # {'index': maxi, 'N': np.round(nt), 'means': miuit}


k = 11
N = 20000
variance = 4 # known
pvar = [10 for i in range(0,k)] # true var of mean parameter is 0.25

# pmiu = KGgenPriorMu()
pmiu = [0.5 for i in range(0,11)]

outputs = []
mistake = []
total = 0
for i in range(0,100):
	if (i+1)%10 == 0:
		print(i)
	outputs.append( KG(k, N, variance, pmiu, pvar) )
	if outputs[i]['index'] != 10:
		mistake.append(outputs[i]['index'])
	total += outputs[i]['total']
print("mistake:"+str(len(mistake))+"times  "+str(mistake))
print("budget"+str(total/100))