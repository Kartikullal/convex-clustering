import numpy as np
import pandas as pd
import warnings
from scipy.stats import multivariate_normal
warnings.filterwarnings("ignore")

class KMeans():
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric
        if(self.metric == 'cosine'):
            self.order = -1
        else:
            self.order = 1

    def get_centroids(self):
        idx = np.random.choice(len(self.X_train), self.k, replace=False)
        centroids = self.X_train[idx, :]
        return centroids
    
    def get_distance(self,A, B):
        if(self.metric == 'euclidian'):
            A_norm = np.square(np.linalg.norm(A, axis = 1)).reshape(A.shape[0],1) * np.ones(shape=(1,B.shape[0]))
            B_norm = np.square(np.linalg.norm(B, axis = 1)) * np.ones(shape=(A.shape[0],1))
            A_M_B = np.matmul(A,B.T)
            d_matrix = A_norm + B_norm - 2*A_M_B
            d_matrix = np.sqrt(d_matrix)

        if(self.metric == 'cosine'):
            from sklearn.metrics.pairwise import cosine_similarity
            d_matrix = cosine_similarity(A,B)

        return d_matrix
    
    def fit(self, X_train):
        self.X_train = X_train
        centroids = self.get_centroids()
        distance = self.get_distance(X_train, centroids)
        
        if(self.metric == 'cosine'):
            points = np.argmax(distance, axis = 1)
        else:
            points =  np.argmin(distance, axis = 1)
            
        old_centroids = centroids
        new_centroids = []
        
        for idx in range(self.k):
            temp_cent = X_train[points==idx].mean(axis=0) 
            new_centroids.append(temp_cent)

        new_centroids = np.nan_to_num(np.vstack(new_centroids))
        distance = self.get_distance(X_train, new_centroids)
        if(self.metric == 'cosine'):
            points = np.argmax(distance, axis = 1)
        else:
            points =  np.argmin(distance, axis = 1)
        while((old_centroids - new_centroids).sum() != 0):
            old_centroids = new_centroids
            new_centroids = []
            for idx in range(self.k):
                temp_cent = X_train[points==idx].mean(axis=0) 
                new_centroids.append(temp_cent)

            new_centroids = np.nan_to_num(np.vstack(new_centroids))

            distance = self.get_distance(X_train, new_centroids)
            
            if(self.metric == 'cosine'):
                points = np.argmax(distance, axis = 1)
            else:
                points =  np.argmin(distance, axis = 1)
        
        return points 


np.random.seed(41)
class GMM:
    def __init__(self, k, max_iter = 100):
        self.k = k
        self.max_iter = max_iter
    
    def fit(self, X):
        #Initialize 
        X = np.array(X)
        self.m, self.n = self.shape = X.shape
        
        self.pi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights_ = np.full( shape=self.shape, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=self.m, size=self.k)
        self.mean_ = np.array([X[row_index,:] for row_index in random_row ])
        self.covariances_ = np.array([ np.cov(X.T) for _ in range(self.k)] )
        
        for iteration in range(self.max_iter):
            # E step
            likelihood = np.zeros( (self.m, self.k) )
            for i in range(self.k):
                distribution = multivariate_normal(mean=self.mean_[i], cov=self.covariances_[i], allow_singular=True)
                likelihood[:,i] = distribution.pdf(X)


            numerator = likelihood * self.pi

            denominator = numerator.sum(axis=1)[:, np.newaxis]

            self.weights_ = numerator / denominator

            self.weights_[np.isnan(self.weights_)] = 0

            
            # M Step
            self.pi = self.weights_.mean(axis=0)
            for i in range(self.k):
                weight = self.weights_[:, [i]]

                total_weight = np.sum(weight)

                self.mean_[i] = np.nan_to_num((X * weight).sum(axis=0) / total_weight)
                self.covariances_[i] = np.nan_to_num(np.cov(X.T, 
                    aweights=(weight/total_weight).flatten(), 
                    bias=True))
            
                
    def predict(self, X):
        m,n = X.shape
        likelihood = np.zeros( (m, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mean_[i], cov=self.covariances_[i], allow_singular=True)
            likelihood[:,i] = distribution.pdf(X)
        numerator = likelihood * self.pi

        denominator = numerator.sum(axis=1)[:, np.newaxis]

        weights = numerator / denominator
        weights[np.isnan(weights)] = 0
        return np.max(weights, axis = 1)
