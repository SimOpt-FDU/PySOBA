import numpy as np

####### Inventory #######
'''
def sS(x):
    littleS = np.ceil(x/40)
    bigS = littleS + x - (littleS -1)*40
    return {'littleS':littleS, 'bigS':bigS, 'Diff': bigS-littleS}
'''

def sSInvt(x, n=1, RandomSeed=-1):
    # simulates the (s,S) inventory example of Koenig & Law
    # x in {1,2,...,1600} is the system index
    # n = number of replications
    # RandomSeed sets the initial seed
    # output is average cost for 30 periods
    littleS = np.ceil(x/40)
    bigS = littleS + x - (littleS -1)*40
    if RandomSeed > 0:
        np.random.seed( RandomSeed )
    Y = np.zeros(n)
    for j in range(0,n):
        InvtPos = bigS
        Cost = 0
        for period in range(0,30):
            Demand = np.random.poisson(lam=25)
            if InvtPos < littleS:
                INext = bigS
                Cost = Cost + 32 + 3*(bigS-InvtPos)
            else:
                INext = InvtPos
            if INext-Demand >= 0:
                Cost = Cost + INext - Demand
            else:
                Cost = Cost + 5*(Demand - INext)
            InvtPos = INext -Demand
        Y[j] = Cost/30
    return (-Y).tolist()



######### SAN ###########

def SAN(x=1, n=1, RandomSeed=-1):
    # Simulation of stochastic activity network
    # with given Means for the exponential activity times
    # x in {1,2,3,4,5} is the system index
    # n = number of replications
    # RandomSeed sets the initial seed
    # output is -time to complete network
	means = [[0.5,1,1,1,1], [1,0.5,1,1,1], [1,1,0.5,1,1],
	 [0.3,1,1,0.4,1,], [1,1,1,1,0.5]]
	if (RandomSeed >0):
		np.random.seed( RandomSeed )
	Y = np.zeros(n)
	for j in range(0,n):
		A = np.random.exponential( scale= means[x], size= 5 ) # means[:,k-1] means the kth column  
		# scale: 1/lambda, so scale is just mean; size:(n,5). 5*n oberservations in total
		# Note: 'size=(n:5)' will return a 2-d matrix [[x1, x2, x3, x4, x5]], here we use 'size=5' to return a vector. 
		Y[j] = max( A[0]+A[4], A[0]+A[2]+A[4], A[1]+A[4] )
	return (-Y).tolist()

############## Normal ###################

def Normal(x, n=1, RandomSeed=-1):
    # normally distributed data
    # x in {1,2,...,11} is the system index
    # n = number of replications
    # RandomSeed sets the initial seed
	mu = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	sigma = 2
	if RandomSeed>0:
		np.random.seed( RandomSeed )
	rand_data = np.random.normal(mu[x], sigma, n)
	return rand_data.tolist()


############# M/M/1 #####################

def MM1(x, n=1, RandomSeed=-1):
    # Simulation of M/M/1 queue with costs for
    # service rate and waiting; arrival rate is 1
    # x in {1,2,...,100} is the system index
    # n = number of replications
    # RandomSeed sets the initial seed
    # output is negative of cost
    mu = [i/5 + 1 for i in range(1,101)]
    if RandomSeed>0:
        np.random.seed( RandomSeed )
    Y = np.zeros(n)
    Wq = 1/(mu[x]*(mu[x]-1))
    for j in range(0, n):
        W = np.zeros(1000)
        D = Wq
        S = np.random.exponential(scale=1/mu[x])
        Snext = np.random.exponential(scale=1/mu[x])
        for i in range(0,1000):
            D = max(0,D + S - np.random.exponential(scale=1))
            W[i] = D + Snext
            S = Snext
            Snext = np.random.exponential(scale=1/mu[x])
        Y[j] = mu[x] + 36 * np.mean(W)
    return (-Y).tolist()


########## TTF ############

def TTF(x, n=1, RandomSeed=-1):
    # function to simulate CTMC TTF example
    # x in {1,2,3,4} is the system index
    # n is number of replications
    # RandomSeed sets the initial seed
    # output is time to system failure
    lam = [1, 1, 1.1, 1.1]
    mu = [9000, 10000, 10000, 11000]
    if RandomSeed>0:
        np.random.seed( RandomSeed )
    Y = np.zeros(n)
    for j in range(0,n):
        ttf = 0
        state = 2
        while state!=0:
            if state==2:
                ttf = ttf - np.log(1-np.random.uniform(size=1))/lam[x]
                state = 1
            else:
                repair = -np.log(1-np.random.uniform(size=1))/mu[x]
                fail = -np.log(1-np.random.uniform(size=1))/lam[x]
                if repair<fail:
                    ttf = ttf + repair
                    state = 2
                else:
                    ttf = ttf +fail
                    state = 0
        Y[j] = ttf
    return Y.tolist()

'''
def ETTF(lambda, mu):
    return (2*lambda + mu)/lambda^2
'''
