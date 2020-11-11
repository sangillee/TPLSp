import numpy as np
from numpy.linalg import norm
from scipy.stats import rankdata
import warnings

class TPLS:
    
    def __init__(self, X, Y, NComp = 50, W = np.ones(1), nmc=0): 
        """ Thresholded Partial Least Squares for Python. - Arthur Lee
            Arguments:
            X -- N-by-p matrix of brain image where each row is one observation and each column is a voxel
            Y -- N-by-1 array of prediction targets. Can be choice, ratings, bids, etc.
            NComp -- Number of PLS components to use in this TPLS model. You can choose to use less than NComp later, but not more.
            W -- N-by-1 observation weights. Should be all positive
            nmc -- switch for skipping mean centering (nmc: no mean centering). This can be done to save RAM space if X and Y are already mean centered
        """

        # Setting observation weights
        n, v = X.shape # number of observations and number of variables, respectively
        if len(W) != n:
            print('Using equal observation weights')
            W = np.ones((n,1)) # if the length of weight does not match the number of rows on X, give default weights
        assert (np.all(W >= 0)), "all observation weights should be positive"
        W = W/sum(W) # weights should be normalized to sum to 1
        
        # Mean-center variables as needed by SIMPLS algorithm
        self.NComp = NComp; self.W = W; self.MtrainX = W.T @ X; self.MtrainY = W.T @ Y # Means of variables
        if nmc == 0: # if no switch is given to skip mean centering
            X = X-self.MtrainX; Y = Y-self.MtrainY
        else:
            print('Mean centering disabled')
            if np.mean(np.abs(self.MtrainX)) > 1e-04:
                warnings.warn('X does not seem to be mean-centered. Results may not be valid')
        
        # allocate memories for output variables, interim variables, and calculate often used variables
        self.scoreCorr = np.zeros(NComp); self.betamap = np.zeros((v,NComp)); self.threshmap = 0.5 * np.ones((v,NComp)) # output variables
        B = np.zeros((NComp,1)); P2 = np.zeros((n,NComp)); C = np.zeros((v,NComp)); sumC2 = np.zeros((v,1)); r = Y; V = np.zeros((v,NComp)) # interim variables
        WYT = (W*Y).T; WT = W.T; WTY2 = WT @ (Y*Y); W2 = W*W # often-used variables

        # Perform Arthur-modified SIMPLS algorithm
        Cov = (WYT@X).T # initial weighted covariance between X and Y
        for i in range(NComp): # starting loop
            print('Calculating Comp #'+str(i+1))
            P = X@Cov; norm_P = np.sqrt(WT@(P*P)); P = P / norm_P # normalized component
            B[i] = (norm(Cov)**2) / norm_P # normalized regression coefficient
            C[:,i] = Cov.flatten()/norm_P # normalized back-projection coefficient

            # Update the orthonormal basis with modified Gram Schmidt
            vi = ((W*P).T @ X).T # weighted covariance between X and current component
            if i != 0:
                vi = vi - V[:,:i] @ (V[:,:i].T @ vi) # orthogonalize vi with regards to previous vis
            vi = vi/ norm(vi); V[:,i] = vi.flatten() # add the normalized vi to orthonormal basis matrix
            Cov = Cov - vi @ (vi.T @ Cov); Cov = Cov - V[:,:(i+1)] @ (V[:,:(i+1)].T @ Cov) # remove the effect of current and previous covariances

            # Back-projection
            self.betamap[:,i] = (C[:,:(i+1)] @ B[:(i+1)]).flatten() # back-projection of coefficients
            sumC2 = sumC2 + (C[:,i]**2).reshape(v,1); P2[:,i] = (P**2).flatten(); r = r - P*B[i] # update interim variables
            if i != 0: # no need to calculate threshold for first component
                se = np.sqrt( P2[:,:(i+1)].T @ (W2*(r**2)) ) # Huber-White Sandwich estimator (assume no small T bias)
                self.threshmap[:,i] = np.abs( (C[:,:(i+1)] @ (B[:(i+1)]/se))/np.sqrt(sumC2) ).flatten() # absolute value of back-projected z-statistics
                self.threshmap[:,i] = (v-rankdata(self.threshmap[:,i]))/v # Convert the absolute z-statistics to rank and normalize to between 0 and 1
        self.pctVar = (B**2)/WTY2 # Compute the percent of variance of Y each component explains
        self.scoreCorr = np.sqrt(self.pctVar)

    def makePredictor(self,compval,threshval):
        # extract betamap from a TPLS model at a given number of components and at given threshold value
        assert (len([threshval])==1), "only one threshold value should be used"
        betamap = self.betamap[:,compval-1]
        if threshval < 1:
            betamap = betamap * (self.threshmap[:,compval-1] <= threshval)
        bias = self.MtrainY - self.MtrainX @ betamap # post-fitting of bias (intercept)
        return betamap, bias
    
    def predict(self,compval,threshval,testX):
        assert (len([threshval])==1), "only one threshold value should be used"
        if threshval == 0:
            score = self.MtrainY * np.ones((len(testX),len(compval)))
        else:
            threshbetamap,bias = self.makePredictor(compval,threshval)
            score = bias + testX @ threshbetamap
        return score