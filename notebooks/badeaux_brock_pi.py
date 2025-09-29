import numpy as np
import matplotlib.pyplot as plt
from math import pi

#Monte Carlo pi estimator
def estimate_pi(n):
    #Create random points within unit square
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)

    #Get sum of all points where the distance from center 
    # is less than 1 (inside unit circle)
    r = np.sum(np.sqrt((x)**2+(y)**2)<1)


    return 4 * r / n

n_range = np.logspace(2, 8, num = 100, base=10, dtype = 'int64')
pi_est_list = np.zeros((100,), dtype=np.float64)
error_list = np.zeros((100,), dtype=np.float64)

for i, n in enumerate(n_range):
    pi_est = estimate_pi(n)
    pi_est_list[i] = pi_est
    error_list[i] = abs(pi_est - pi)

#Plotting results
plt.plot(n_range, pi_est_list , label='Estimated Pi')
plt.axhline(y=pi, color='r', linestyle='--', label='Actual Pi')
plt.xscale('log')
plt.xlabel('Number of Samples (n)')
plt.ylabel('Estimated Pi Value')
plt.title('Monte Carlo Pi Estimation')
plt.legend()
plt.grid()

def RSamplesOfPi(r, n):
    rOfN = np.zeros((r,))
    for i in range(r):
        rOfN[i] = estimate_pi(n)
    return rOfN

sample1 = np.array(RSamplesOfPi(500,10**3))
sample2 = np.array(RSamplesOfPi(500,10**4))
sample3 = np.array(RSamplesOfPi(500,10**5))



print(f"Sample 1 Mean: {np.mean(sample1)}, 1/sqrt(n): {1/np.sqrt(10**3)} Std Dev: {np.std(sample1)}")
print(f"Sample 2 Mean: {np.mean(sample2)}, 1/sqrt(n): {1/np.sqrt(10**4)} Std Dev: {np.std(sample2)}")
print(f"Sample 3 Mean: {np.mean(sample3)}, 1/sqrt(n): {1/np.sqrt(10**5)} Std Dev: {np.std(sample3)}")

plt.hist(sample1, bins=23, alpha=0.5, label='n=10^3')
plt.hist(sample2, bins=23, alpha=0.5, label='n=10^4')
plt.hist(sample3, bins=23, alpha=0.5, label='n=10^5')

#plt.axvline(x=pi, color='r', linestyle='--', label='Actual Pi')
plt.xlabel('Estimated Pi Value')
plt.ylabel('Frequency')
plt.title('Distribution of Pi Estimates for Different Sample Sizes')
plt.legend()
plt.show()
