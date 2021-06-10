from Sampler import *
import numpy as np
import scipy.stats as st
import math

def mean(nlist):
	return np.mean(nlist)
def var(nlist):
	return np.var(nlist, ddof=1)

# Choose a sampler from Sampler.py
def genSample(k=1, n=1, RandomSeed=-1):
	return Normal(x=k, n=int(n), RandomSeed=RandomSeed)

def OCBA(k, n0, N, τ): 
	index = [x for x in range(0,k)]
	samples = []
	means = np.zeros(k)
	variances = np.zeros(k)
	dik = np.zeros(k) # the difference of means between system i and maxi
	for i in range(0,k):
		sample_i=genSample(i, n0)
		samples.append(sample_i)
	t = 0
	nt = np.array( [n0 for i in range(0,k)] )
	nt1 = np.zeros(k)
	bt = sum( nt )
	while bt<N:
		for i in range(0,k):
			means[i] = mean(samples[i])
			variances[i] = var(samples[i])
		maxi = np.argmax(means)
		for i in range(0,k):
			dik[i] = means[maxi] - means[i]
		bt = bt + τ
		# calculate nt1
		term1 = sum( variances[i] / pow(dik[i], 2) for i in range(0,k) if i!=maxi ) # note: dik[maxi] = 0
		term2 = [variances[i]/pow(dik[i],4) for i in range(0,k) if i!=maxi]
		term3 = np.sqrt( variances[maxi] * (sum(term2)) )
		r = bt / (term1 + term3)
		for i in range(0,k):
			if i==maxi:
				nt1[maxi] = term3 * r
			else:
				nt1[i] = variances[i] * r / pow(dik[i], 2)
		new_n = np.zeros(k)
		for i in range(0,k):
			new_n[i] = max(0, math.ceil(nt1[i] - nt[i]))
			new_sample = genSample(i, int(new_n[i]))	
			samples[i] = samples[i] + new_sample
		t = t+1
		nt = nt + new_n # update the allocation
	return [maxi, nt, means] # {'index': maxi,'N': nt, 'means': means}

def get_nt1(k, maxi, bt, variances, lambdaik, dik, nt, survive):
	nt1 = [nt[i] for i in range(0,k)]
	if sum(survive) == 1:  # only one system survived
		remain = survive.index(True)
		nt1[remain] = bt - sum(nt) + nt[remain]
		return nt1	

	eta = np.zeros(k)
	NSbt = 0 # budget for systems not survived
	for i in range(0,k):
		if i!=maxi and survive[i]:
			term1 = math.sqrt(lambdaik[i]) * (nt[i] -1 + lambdaik[i] * pow(dik[i], 2)) / (nt[i]-2)
			eta[i] = term1 * st.t.pdf( math.sqrt(lambdaik[i]) * dik[i], nt[i]-1) # student-t 
		if not survive[i]:
			NSbt += nt[i]
	if survive[maxi]:
		eta[maxi] = sum([eta[i] for i in range(0,k) if i != maxi] )
	# calculate nt1
	term2 = pow(variances * eta, 0.5) # note: variances and eta are both k-dim vector
	r = (bt - NSbt) / sum(term2)
	for i in range(0,k):
		if survive[i]:
			nt1[i] = r * term2[i] # k-dim
	return nt1 #nt1 is an array

def EVI(k, n0, N, τ): 
	index = [x for x in range(0,k)]
	samples = []
	means = np.zeros(k)
	variances = np.zeros(k)
	for i in range(0,k):
		sample_i=genSample(i, n0)
		samples.append(sample_i)
	t = 0
	nt = np.array( [n0 for i in range(0,k)] )
	nt1 = np.zeros(k)
	bt = sum( nt )
	while bt<N:
		for i in range(0,k):
			means[i] = mean(samples[i])
			variances[i] = var(samples[i])
		survive = [True for i in range(0,k)] # L
		maxi = np.argmax(means)
		lambdaik = [ 1/(variances[i]/nt[i]+variances[maxi]/nt[maxi]) for i in range(0,k) ]
		dik = [ means[maxi] - means[i] for i in range(0,k)] # the difference of means between system i and maxi
		bt = bt + τ
		nt1 = get_nt1(k, maxi, bt, variances, lambdaik, dik, nt, survive)
		ndelta = np.zeros(k)
		for i in range(0,k):
			ndelta[i] = nt1[i]-nt[i]
			if ndelta[i]<0:
					survive[i] = False
					nt1[i] = nt[i]
		while(min(ndelta)<0):
			for i in range(0,k):
				if not survive[maxi]:
					lambdaik[i] = nt[i] / variances[i]
				else:
					if not survive[i]:
						lambdaik[i] = nt[maxi] / variances[maxi] # symmetrical elements
					else:
						lambdaik[i] = 1 / (variances[i]/nt[i] + variances[maxi]/nt[maxi])
						# symmetrical elements
			nt1 = get_nt1(k, maxi, bt, variances, lambdaik, dik, nt, survive)
			# update ndelta
			for i in range(0,k):
				ndelta[i] = nt1[i]-nt[i]
				if ndelta[i]<0:
					survive[i] = False
					nt1[i] = nt[i]
		new_n = np.zeros(k)
		for i in range(0,k):
			new_n[i] = math.ceil(nt1[i] - nt[i])
			new_sample = genSample(i, int(new_n[i]))	
			samples[i] = samples[i] + new_sample
		t = t+1
		nt = nt + new_n # update the allocation
	return [maxi, nt, means] # {'index': maxi,'N': nt, 'means': means}


'''
result = [[], []]
mistake = [[], []]
filename = ['4.txt','5.txt']
for i in range(0,100):
	if (i+1)%10 == 0:
		print(i)
	output = OCBA(k = 11, n0 = 10, N = 5000,  τ = 100)
	result[0].append(str(output))
	if output[0] != 10:
		mistake[0].append(output[0])
	
	output = EVI(k = 11, n0 = 10, N = 5000,  τ = 100)
	result[1].append(str(output))
	if output[0] != 10:
		mistake[1].append(output[0])


print("Mistake decisions:")
print("times: 'OCBA':"+str(len(mistake[0]))+";'EVI':"+str(len(mistake[1])))
print( {'OCBA':mistake[0], 'EVI':mistake[1]} )

result1 = "\n".join(result[0])
with open(filename[0],'w') as f:
    f.write(result1)
f.close()

result2 = "\n".join(result[1])
with open(filename[1],'w') as f:
    f.write(result2)
f.close()
'''
