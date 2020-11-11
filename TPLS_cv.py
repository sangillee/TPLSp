import numpy as np
from TPLS import TPLS

class TPLS_cv:
    
    def __init__(self, X, Y, CVfold, NComp = 50, W = np.ones(1)): 
        """ TPLS cross-validation trainer
            Arguments:
            X -- N-by-p matrix of brain image where each row is one observation and each column is a voxel
            Y -- N-by-1 vector of prediction targets. Can be choice, ratings, bids, etc.
            CVfold -- N-by-1 vector of CV fold indices (1, 2, 3, ...)
            W -- N-by-1 observation weights. Should be all positive
        """

        # Setting observation weights
        n, v = X.shape # number of observations and number of variables, respectively
        if len(W) != n:
            print('Using equal observation weights')
            W = np.ones((n,1)) # if the length of weight does not match the number of rows on X, give default weights
        assert (np.all(W >= 0)), "all observation weights should be positive"
        
        uniqfold = np.unique(CVfold)
        self.NComp = NComp
        self.numfold = len(uniqfold)
        self.testfold = np.zeros((n,1))
        self.cvMdls = []
        for i in range(self.numfold): # starting loop
            print('Fold #'+str(i+1))
            train = CVfold != uniqfold[i]
            self.testfold[~train] = i
            self.cvMdls.append(TPLS(X[train.flatten(),:],np.atleast_2d(Y[train]).T,NComp,np.atleast_2d(W[train]).T))