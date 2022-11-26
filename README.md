# convex-clustering

Implemented Kmeans and Gaussian Mixture models from scratch. Created an evaluation function to calculate purity and gini value for the clustered labels.

Fit the Kmeans model on MNIST images (0-9), and clustered with a purity of 0.6354833333333333 and gini value of 0.6102175291133967. Can get better results for both if dimensionality reduction is performed on the training data.

Fit the GMM model on Spambase dataset. Fit two separate GMM models for all positive lables and negative labels in spambase, and caculated likelihood by multiplying the values with the prior probability for each class. Then predicted the right class with maximum liklihood. The result was an accuracy of 68%.
 
