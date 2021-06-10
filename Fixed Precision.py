from Sampler import *
import numpy as np
import scipy.stats as st

def mean(nlist):
	return np.mean(nlist)
def var(nlist):
	return np.var(nlist, ddof=1)

# Choose a sampler from Sampler.py
def genSample(k=1, n=1, RandomSeed=-1):
	return SAN(x=k, n=int(n), RandomSeed=RandomSeed)

def Rinotth(k=1, n0=1, pstar=0.95, conf=0.99, rep=10000):
	# function to return estimate of Rinott h and upper confidence bound on it
	# pstar = 1-alpha value (PCS)
	# rep = number of replications to use for estimate
	# conf = confidence level on upper bound
	Z = np.random.normal(loc=0, scale=1, size=(rep, k-1))
	Y = np.random.chisquare(df=n0-1, size=(rep, k-1)) # ncol=k-1
	C = np.random.chisquare(df=n0-1, size=rep ) # array, not matrix. (ncol=rep)
	Cmat = np.mat( [C for i in range(0,k-1)] ).T # nrow = rep, ncol= k-1
	denom = np.sqrt( (n0-1) * (1/Y + 1/Cmat))
	Zdenom = np.multiply(Z, denom) # nrow = rep
	Hrep = np.zeros(rep)
	for i in range(0,rep):
		rowmax=float('-inf')
		for j in range(0,k-1):
			if Zdenom[i,j] > rowmax:
				rowmax = Zdenom[i,j]
		Hrep[i] = rowmax
	Hrep.sort()
	Hstar = np.percentile(Hrep, 100 * pstar) # quantile range from 0 to 100
	#upper = np.ceil(pstar * rep + qnorm(conf) * sqrt(pstar * (1. - pstar) *rep) + 0.5)
	upper = np.ceil(pstar * rep + st.norm.ppf(conf) * np.sqrt(pstar * (1-pstar) * rep) +0.5 )
	Hupper = Hrep[int(upper)]
	return {'h':Hstar, 'UCB':Hupper}

def RinottProcedure(k=1, n0=2, alpha=0.05, delta=0):
	#output is a dict:
	# { 'index': index of the system with the largest sample mean,
	# 'sample_means': sample means of each system,
	# 'sample_vars': sample variances of each system,
	# 'N': number of observations generated for each system }
	#"Step 1"
	Rh = Rinotth(k, n0, pstar= 1-alpha, conf=0.99, rep=10000) # Rh is a dict
	h = Rh['UCB']
	samples=[]
	sample_vars=[]
	for i in range(0,k):
		sample_i=genSample(i, n0)
		var_i=var(sample_i)
		sample_vars.append(var_i)
		samples.append(sample_i)
	
	#"Step 2"
	nmax=[]
	sample_means=[]
	for i in range(0,k):
		nmax_i=np.ceil(h**2 * sample_vars[i]**2 / (delta**2) )
		if nmax_i > n0:
			nmax.append(nmax_i)
			samples[i].extend(genSample(i, nmax_i-n0))
			mean_i=mean(samples[i])
			sample_means.append(mean_i)
		else:
			nmax.append(n0)
			mean_i=mean(samples[i])
			sample_means.append(mean_i)
	
	#"find the max sample mean"
	index = np.argmax(sample_means)
	total = sum(nmax)
	return [index, total, nmax, sample_means] # {'index':index, 'total_budget': total, 'N':nmax, 'sample_means':sample_means}

def KN(k, alpha, n0, delta, seed=-1):
	index=[x for x in range(0,k)]
	survive=[True for x in range(0,k)]
	samples=[]
	sums=[]
	for i in range(0,k):
		if seed>0:
			sample_i=genSample(i, n0, seed)
		else:
			sample_i=genSample(i, n0)
		samples.append(sample_i)
		sums.append(sum(sample_i))
	sample_cov=np.cov(samples)
	# sample_corr = np.corrcoef(samples)
	# print(sample_corr)  "see the correlation"

	eta = 0.5*( (2*alpha/(k-1))**(-2/(n0-1)) -1)
	hsq = 2*eta*(n0-1)
	Elim=[0 for i in range(0,k)]
	r=n0
	while(len(index)>1):
		r=r+1
		if seed>0:
			seed = seed + 1
		for i in index:
			if seed>0:
				newSample=genSample(i, 1, seed)
			else:
				newSample=genSample(i, 1)
			sums[i] = sums[i] + newSample[0] # newSample is a list
		for i in index:
			for l in index:
				Ssq=sample_cov[i][i]+ sample_cov[l][l]- 2*sample_cov[i][l]
				W=max(0, (delta/2) * (hsq * Ssq / delta**2 - r))
				if sums[i] < sums[l]-W :
					survive[i]=False
					Elim[i]=r
					break
		index=[i for i in index if survive[i]]
	Elim[index[0]] = r
	total = sum(Elim)
	sample_means = [sums[i] / Elim[i] for i in range(0,k)]
	return [index[0], total, Elim, sample_means] # {'index':index, 'total_budget': total, 'N':Elim, 'sample_means':sample_means}

def IZfree(k, n0, alpha, seed=-1):
	c= -2*np.log(2*alpha/(k-1))
	index=[x for x in range(0,k)]
	survive=[True for x in range(0,k)]
	samples=[]
	means=[]
	sp_diff=[[0 for x in range(0,k)] for y in range(0,k)]
	for i in range(0,k):
		if seed>0:
			sample_i=genSample(i, n0, seed)
		else:
			sample_i=genSample(i, n0)
		means.append(mean(sample_i))
		samples.append(sample_i)
	for i in range(0,k):
		for j in range(0,k):
			sp_diff[i][j]=var( (np.array(samples[i]) - np.array(samples[j])).tolist() )
	n=n0
	Elim=[0 for i in range(0,k)]
	gval=[[0 for x in range(0,k)] for y in range(0,k)]

	while len(index)>1:
		for i in index:
			for j in index:
				if j==i:
					continue
				tij= n/sp_diff[i][j]
				gval[i][j]=np.sqrt( (c+ np.log(tij+1) ) * (tij+1) )
				if tij * (means[i]-means[j]) <= -gval[i][j]:
					survive[i]=False
					Elim[i]=n
					break
		index=[i for i in index if survive[i]]
		if seed>0:
			seed = seed + 1
		for i in index:
			if seed>0:
				new_ob=genSample(i,1,seed)
			else:
				new_ob=genSample(i,1)
			means[i]= means[i]+ (new_ob[0] -means[i])/(n+1)
		n= n+1
	Elim[index[0]] = n-1
	total = sum(Elim)
	return [index[0], total, Elim, means] # {'index':index, 'total_budget': total, 'N':Elim, 'sample_means':means}


'''
# seed = np.random.randint(1, high=1000000, size=100, dtype='l').tolist()  # to get seed
seed=[435912, 535854, 482059, 610400, 69575, 431080, 62180, 266894, 522072, 605668, 
675197, 94045, 217812, 508857, 475609, 934658, 427022, 667193, 484165, 679989, 289157,
590275, 834295, 376645, 626706, 854011, 766228, 180470, 917276, 411731, 384032, 52878, 
540215, 377224, 941756, 408395, 970157, 591205, 926361, 234047, 829020, 443857, 62805, 
380960, 674455, 224555, 998861, 795259, 322639, 447416, 371634, 71645, 334722, 551648, 
500586, 725188, 424570, 902569, 27041, 549712, 424999, 443459, 867693, 261257, 516030, 
348259, 4480, 159134, 364543, 611849, 492989, 31851, 933702, 707862, 368732, 926596, 220964, 
507324, 451481, 279315, 982760, 480750, 670214, 263229, 126648, 556572, 262981, 719940, 
401560, 685723, 446264, 359330, 214700, 334946, 728088, 687510, 489067, 406077, 228409, 540966]

result = [[], [], []]
N = [[], [], []]
mistake = [0,0,0]
filename = ['1.txt','2.txt','3.txt']
for i in range(0,100):
	print(i)
	output = RinottProcedure(k=5, n0=20, alpha=0.05, delta=0.066)
	result[0].append(str(output))
	N[0].append(output[1])
	if output[0] != 3:
		mistake[0] += 1
	
	output = KN(k=5, alpha=0.05, n0=20, delta=0.066, seed=seed[i])
	result[1].append(str(output))
	N[1].append(output[1])
	if output[0] != 3:
		mistake[1] += 1

	output = IZfree(k=5, alpha=0.05, n0=20, seed=seed[i])
	result[2].append(str(output))
	N[2].append(output[1])
	if output[0] != 3:
		mistake[2] += 1

print( {'Rinott':mean(N[0]), 'KN':mean(N[1]), 'IZfree':mean(N[2])} )
print("Times of mistake:")
print(mistake)
result1 = "\n".join(result[0])
with open(filename[0],'w') as f:
    f.write(result1)
f.close()

result2 = "\n".join(result[1])
with open(filename[1],'w') as f:
    f.write(result2)
f.close()

result3 = "\n".join(result[2])
with open(filename[2],'w') as f:
    f.write(result3)
f.close()
'''