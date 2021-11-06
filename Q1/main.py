import numpy as np
from math import log, pi

def discriminant(x, mean, covariance, dimension, probability):
    #Check if it is univariate:numpy.linalg.LinAlgError: 1-dimensional array given. Array must be at least two-dimensional
    if dimension == 1: 
        dis = (-0.5*(x - mean) * (1 / covariance))* (x-mean) - 0.5*log(2*pi) - 0.5*log(covariance) 
    else: 
        dis = np.matmul(np.matmul(-0.5*(x - mean), np.linalg.inv(covariance)), (x-mean).T) -0.5*dimension*log(2*pi) - 0.5*log(np.linalg.det(covariance))
    if(probability == 0):
        return dis
    else: 
        dis += log(probability)
    return dis

def main():
    # Given Data
    dataclass = [
            [[-5.01, -8.12, -3.68], [-5.43, -3.48, -3.54], [1.08, -5.52, 1.66], [0.86, -3.78, -4.11], [-2.67, 0.63, 7.39], [4.94, 3.29, 2.08], [-2.51, 2.09, -2.59], [-2.25, -2.13, -6.94], [5.56, 2.86, -2.26], [1.03, -3.33, 4.33]],
            [[-0.91, -0.18, -0.05], [1.30, -2.06, -3.53], [-7.75, -4.54, -0.95], [-5.47, 0.50, 3.92], [6.14, 5.72, -4.85], [3.60, 1.26, 4.36], [5.37, -4.63, -3.65], [7.18, 1.46, -6.66], [-7.39, 1.17, 6.30], [-7.50, -6.32, -0.31]],
            [[5.35, 2.26, 8.13], [5.12, 3.22, -2.66], [-1.34, -5.31, -9.87], [4.48, 3.42, 5.19], [7.11, 2.39, 9.21],[7.17, 4.33, -0.98], [5.75, 3.97, 6.65], [0.77, 0.27, 2.41], [0.90, -0.43, -8.71], [3.52, -0.36, 6.43]] 
        ]
    dataclasses = [np.array(l) for l in dataclass]    
    #Find other values required for computing the values: 
    n = len(dataclasses)            # number of classes 
    d = len(dataclasses[0][0])      # number of features

    #Assuming 
    probability = [0.25, 0.5, 0.25]

    #Find mean and covariance    
    means = []                      # d-component mean vector
    covariance = []                 # d by d covariance matrix for each set
    
    #Finding means in each column
    for sing in dataclasses:
        means.append(sing.mean(axis=0))
        covariance.append(np.cov(sing.T))
    k = 0
    for dataclass in dataclasses:
        k+=1
        print("The following data should be classified as: ", k)
        missed = 0
        count = 0
        for data in dataclass:
            gi = [0] * n           # each gi 
            for i in range(n):
                gi[i] = discriminant(data, means[i], covariance[i], d, probability[i])
                
            #find maximum g[i]
            maximum_indices = gi.index(max(gi)) + 1
            count+=1
            if(maximum_indices != k):
                missed += 1
            print(data, "\t classified as: \t", maximum_indices )
        print("Success: \t", ((count - missed) / count)*100 , "%")
        print("Failure: \t", ((missed) / count)*100 , "%")
            

if __name__ == '__main__':
    main()
    
