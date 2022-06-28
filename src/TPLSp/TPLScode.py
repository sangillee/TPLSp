import numpy as np
import math
from scipy.stats import rankdata, spearmanr, pearsonr
import matplotlib.pyplot as plt
import warnings

class TPLS:
    
    def __init__(self, X, Y, NComp = 25, W = None, nmc = 0): 
        """ Constructor method for fitting a T-PLS model with given data X and Y.
            X       Numerical numpy matrix of predictors. Typically single-trial betas where each column is a voxel and row is observation
            Y       Numerical numpy column vector to predict. Binary 0 and 1 in case of classification, continuous variable in case of regression
            NComp   (Optional) Number of PLS components to compute. Default is 25.
            W       (Optional) Observation weights. By default, all observations have equal weight.
            nmc     (Optional) 'no mean centering'. Default is 0. If 1, T-PLS will skip mean-centering.
                    This option is only provided in case you already mean-centered the data and want to save some memory usage.
        """

        # input checking
        TPLSinputchecker(X,'X','mat',None,None,1); n, v = X.shape
        Y = TPLSinputchecker(Y,'Y','colvec',None,None,1)
        TPLSinputchecker(NComp,'NComp','scalar',None,1,0,1)
        if W is None: W = np.ones((n,1))
        W = TPLSinputchecker(W,'W','colvec',None,0); W = W/sum(W); # normalize weight sum to 1
        TPLSinputchecker(nmc,'nmc','scalar')
        assert(n==Y.size and n==W.size),'X, Y, and W should have equal number of rows'
        
        # Mean-center variables as needed by SIMPLS algorithm
        self.NComp = NComp; self.MX = W.T @ X; self.MY = W.T @ Y # calculating weighted means of X and Y
        if nmc == 0: # do mean centering
            X = X-self.MX; Y = Y-self.MY # subtract means
        elif np.any(np.abs(self.MX) > 1e-04):
            warnings.warn('Skipped mean centering, but X does not seem to be mean-centered. Results may be invalid')
        
        # allocate memories
        self.pctVar = np.zeros((NComp,1)); self.scoreCorr = np.zeros((NComp,1)); # percent of variance of Y each component explains, weighted correlation between Y and current component
        self.betamap = np.zeros((v,NComp)); self.threshmap = 0.5 * np.ones((v,NComp)); self.zmap = np.zeros((v,NComp)) # output variables
        B = np.zeros((NComp,1)); P2 = np.zeros((n,NComp)); C = np.zeros((v,NComp)); sumC2 = np.zeros((v,1)); r = Y; V = np.zeros((v,NComp)) # interim variables
        WT = W.T; WTY2 = WT @ (Y*Y); W2 = W*W # often-used variables in calculation

        # Perform Arthur-modified SIMPLS algorithm
        Cov = (((W*Y).T)@X).T; normCov = np.linalg.norm(Cov) # initial weighted covariance between X and Y
        for i in range(NComp): # starting loop
            print('Calculating Comp #'+str(i+1))
            P = X@Cov; norm_P = np.sqrt(WT@(P*P)); # this is the component and its weighted stdev
            P = P / norm_P; B[i] = (normCov**2) / norm_P; C[:,i] = Cov.flatten()/norm_P # normalize component, beta, and back-projection coefficient
            self.pctVar[i] = (B[i]*B[i])/WTY2; self.scoreCorr[i] = np.sqrt(self.pctVar[i])

            # Update the orthonormal basis with modified Gram Schmidt
            vi = ((W*P).T @ X).T # weighted covariance between X and current component
            if i != 0: vi = vi - V[:,:i] @ (V[:,:i].T @ vi) # orthogonalize with regards to previous components
            vi = vi/ np.linalg.norm(vi); V[:,i] = vi.flatten() # add the normalized vi to orthonormal basis matrix
            Cov = Cov - vi @ (vi.T @ Cov); Cov = Cov - V[:,:(i+1)] @ (V[:,:(i+1)].T @ Cov); normCov = np.linalg.norm(Cov) # Deflate Covariance using the orthonormal basis matrix

            # Back-projection
            self.betamap[:,i] = (C[:,:(i+1)] @ B[:(i+1)]).flatten() # back-projection of coefficients
            sumC2 = sumC2 + (C[:,i]**2).reshape(v,1); P2[:,i] = (P**2).flatten(); r = r - P*B[i] # some variables that will facilitate computation later
            if i != 0: # no need to calculate threshold for first component
                se = np.sqrt( P2[:,:(i+1)].T @ (W2*(r**2)) ) # Huber-White Sandwich estimator (assume no small T bias)
                self.zmap[:,i] = ( (C[:,:(i+1)] @ (B[:(i+1)]/se))/np.sqrt(sumC2) ).flatten() # back-projected z-statistics
                self.threshmap[:,i] = (v-rankdata(np.abs(self.zmap[:,i])))/v # convert into thresholds between 0 and 1
        
            # check if there's enough covariance to milk
            if normCov < 10*np.finfo(float).eps:
                print('All Covariance between X and Y has been explained. Stopping...'); break
            elif self.pctVar[i] < 10*np.finfo(float).eps: # Proportion of Y variance explained is small
                print('New PLS component does not explain more covariance. Stopping...'); break

    def makePredictor(self,compval,threshval):
        """ Method for extracting the T-PLS predictor at a given compval and threshval
            input   compval     Vector of number of components to use in final predictor
                                (e.g., [3,5] will give you two betamaps, one with 3 components and one with 5 components
                    threshval   Scalar thresholding value to use in final predictor.
                                (e.g., 0.1 will yield betamap where only 10% of coefficients will be non-zero)
            return  betamap     T-PLS predictor coefficient
                    bias        Intercept for T-PLS model.
        """
        compval = TPLSinputchecker(compval,'compval','rowvec',self.NComp,1,0,1)
        TPLSinputchecker(threshval,'threshval','scalar',1,0)
        if threshval == 0:
            betamap = self.betamap[:,compval-1] * 0
        else:
            betamap = self.betamap[:,compval-1] * (self.threshmap[:,compval-1] <= threshval)
        bias = self.MY - self.MX @ betamap # post-fitting of bias
        return betamap, bias
    
    def predict(self,compval,threshval,testX):
        """ Method for making predictions on a testing dataset testX
            input   compval     Vector of number of components to use in final predictor
                    threshval   Single number of thresholding value to use in final predictor.
                    testX       Data to be predicted. In same orientation as X
            return  score       Prediction scores on a testing dataset
        """
        TPLSinputchecker(testX,'testX')
        threshbetamap,bias = self.makePredictor(compval,threshval)
        score = bias + testX @ threshbetamap
        return score

class TPLS_cv:
    
    def __init__(self, X, Y, CVfold, NComp = 25, W = None, nmc = 0): 
        """ Constructor method for fitting a cross-validation T-PLS model
            X       Numerical matrix of predictors. Typically single-trial betas where each column is a voxel and row is observation
            Y       Variable to predict. Binary 0 and 1 in case of classification, continuous variable in case of regression
            CVfold  Cross-validation testing fold information. Can either be a vector or a matrix, the latter being more general.
                    Vector: n-by-1 vector. Each element is a number ranging from 1 ~ numfold to identify which testing fold eachobservation belongs to
                    Matrix: n-by-numfold matrix. Each column indicates the testing data with 1 and training data as 0.
                    Example: For leave-one-out CV, Vector would be 1:n, Matrix form would be eye(n)
                    Matrix form is more general as it can have same trial be in multiple test folds
            NComp   (Optional) Number of PLS components to compute. Default is 25.
            W       (Optional) Observation weights. Optional input. By default, all observations have equal weight.
                    Can either be a n-by-1 vector or a n-by-nfold matrix where each column is observation weights in that CV fold
            nmc     (Optional) 'no mean centering'. See TPLS for more detail.
        """

        # input checking
        TPLSinputchecker(X,'X','mat',None,None,1); n = X.shape[0]
        Y = TPLSinputchecker(Y,'Y','colvec',None,None,1)
        if CVfold.ndim == 1 or CVfold.shape[1] == 1:
            CVfold = TPLSinputchecker(CVfold,'CVfold','colvec')
        else:
            TPLSinputchecker(CVfold,'CVfold','mat')
        TPLSinputchecker(NComp,'NComp','scalar',None,1,0,1)
        if W is None: W = np.ones((n,1))
        W = np.atleast_2d(W) # could be a vector, could be a matrix
        TPLSinputchecker(W,'W',None,None,0);
        TPLSinputchecker(nmc,'nmc','scalar')
        self.CVfold, self.numfold = self.prepCVfold(CVfold) # convert CVfold into matrix form, if not already
        assert(n==Y.size and n==self.CVfold.shape[0] and n==W.shape[0]),'X, Y, W, and CV fold should have same number of rows'
        if W.shape[1] == 1: W = np.repeat(W,self.numfold,1) # convert into matrix form, if not already
        
        self.NComp = NComp
        self.cvMdls = []
        for i in range(self.numfold):
            print('Fold #'+str(i+1))
            train = self.CVfold[:,i] == 0
            self.cvMdls.append(TPLS(X[train.flatten(),:],Y[train],NComp,W[train,i],nmc))

    @staticmethod
    def prepCVfold(inCVfold):
        """ prepare CV fold data into a matrix form, which is more generalizable """
        if inCVfold.shape[1] == 1: # vector
            uniqfold = np.unique(inCVfold); nfold = len(uniqfold)
            CVfold = np.zeros((inCVfold.shape[0],nfold))
            for i in range(nfold):
                CVfold[:,i] = 1 * np.atleast_2d(inCVfold == uniqfold[i]).T
        elif inCVfold.shape[1] > 1: # matrix
            nfold = inCVfold.shape[1]; CVfold = inCVfold
            if np.any(np.logical_and(CVfold.flatten() != 0, CVfold.flatten() != 1)):
                raise Exception('Non-binary element in matrix form CVfold. Perhaps you meant to use vector form?')
        else:
            raise Exception("unexpected size of CVfold")
        return CVfold, nfold


class evalTuningParam:
    
    def __init__(self, cvmdl, perftype, X, Y, compvec, threshvec, subfold = None): 
        """ Evaluating cross-validation performance of a TPLS_cv model at compvec and threshvec
            cvmdl       A TPLS_cv object
            perftype    CV performance metric type. One of LLbinary, negMSE, Pearson, Spearman, AUC, ACC.
            X           The same X as used in TPLS_cv.
            Y           The same Y as used in TPLS_cv.
            compvec     Vector of number of components to test in cross-validation.
            threshvec   Vector of threshold level [0 1] to test in cross-validation.
            subfold     (Optional) vector of subdivision within testing fold to calculate performance. For example scan run division within subject.
        """

        # input checking
        assert(np.isin(perftype,['LLbinary','negMSE','Pearson','Spearman','AUC','ACC'])), 'Unknown performance metric'; self.type = perftype;
        TPLSinputchecker(X,'X','mat',None,None,1); n, v = X.shape
        Y = TPLSinputchecker(Y,'Y','colvec',None,None,1)
        compvec = TPLSinputchecker(compvec,'compvec','rowvec',cvmdl.NComp,1,0,1); compvec = np.sort(compvec); self.compval = compvec;
        threshvec = TPLSinputchecker(threshvec,'threshvec','rowvec',1,0); threshvec = np.sort(threshvec); self.threshval = threshvec;
        if subfold is None:
            subfold = np.ones((n,1))
        else:
            subfold = TPLSinputchecker(subfold,'subfold','colvec')
        
        # Perform CV prediction and performance measurement
        perfmat = np.empty((len(compvec),len(threshvec),cvmdl.numfold)); perfmat.fill(np.nan)
        for i in range(cvmdl.numfold):
            print('Fold #'+str(i+1))
            testCVfold = cvmdl.CVfold[:,i] == 1
            Ytest = Y[testCVfold]
            testsubfold = subfold[testCVfold]
            uniqtestsubfold = np.unique(testsubfold)
            for j in range(len(threshvec)):
                predmat = cvmdl.cvMdls[i].predict(compvec,threshvec[j].item(),X[testCVfold.flatten(),:])
                smallperfmat = np.empty((len(compvec),len(uniqtestsubfold)))
                for k in range(len(uniqtestsubfold)):
                    subfoldsel = testsubfold == uniqtestsubfold[k]
                    smallperfmat[:,k] = self.util_perfmetric(predmat[subfoldsel.flatten(),:],Ytest[subfoldsel],perftype)
                perfmat[:,j,i] = np.nanmean(smallperfmat, axis=1)
        
        # prepare output object
        self.perfmat = perfmat; avgperfmat = np.nanmean(perfmat, axis=2) # mean performance
        self.perf_best = np.nanmax(avgperfmat).item() # best mean performance
        row_best,col_best = np.where(avgperfmat==self.perf_best) # coordinates of best point
        self.compval_best = compvec[row_best[0]].item(); self.threshval_best = threshvec[col_best[0]].item(); # component and threshold of best point
        standardError = np.nanstd(perfmat[row_best[0],col_best[0]])/np.sqrt(perfmat.shape[2]); # standard error of best point
        candidates = avgperfmat[:,np.arange(col_best[0]+1)] > (self.perf_best-standardError)
        col_1se,row_1se = np.where(candidates.T) # coordinates of 1SE point
        self.perf_1se = avgperfmat[row_1se[0],col_1se[0]].item(); # performance of 1SE point
        self.compval_1se = compvec[row_1se[0]].item(); self.threshval_1se = threshvec[col_1se[0]].item()
        maxroute = np.max(avgperfmat,0); maxrouteind = np.argmax(avgperfmat,0)
        self.best_at_threshold = np.vstack((maxroute,threshvec,compvec[maxrouteind])).T

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        meansurf = np.nanmean(self.perfmat, axis = 2)
        X, Y = np.meshgrid(self.threshval, self.compval)
        ax.plot_surface(X, Y, meansurf, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        ax.set_xlabel('Proportion of Voxels Left')
        ax.set_ylabel('Number of PLS components')
        ax.set_zlabel(self.type)
        ax.scatter(self.threshval_best,self.compval_best,self.perf_best,c='blue')
        ax.scatter(self.threshval_1se,self.compval_1se,self.perf_1se,c='red')
        ax.scatter(self.threshval,self.best_at_threshold[:,2],self.best_at_threshold[:,0],c='orange')
        plt.show()

    @staticmethod
    def util_perfmetric(predmat,testY,perftype):
        if perftype == 'LLbinary':
            assert(np.all(np.logical_or(testY == 0, testY==1))), 'LL binary can be only calculated for binary measures'
            predmat[testY!=1,:] = 1 - predmat[testY!=1,:] # flip probability
            predmat[predmat>1] = 1; predmat[predmat<=0] = np.finfo(float).tiny # take care of probability predictions outside of range
            Perf = np.nanmean(np.log(predmat),0)
        elif perftype == 'negMSE':
            testY = np.atleast_2d(testY).T
            Perf = -np.nanmean((predmat-testY)**2,0)
        elif perftype == 'ACC':
            assert(np.all(np.logical_or(testY == 0, testY==1))), 'Accuracy can be only calculated for binary measures'
            testY = np.atleast_2d(testY).T
            Perf = np.nanmean(testY == 1*(predmat>0.5),axis=0)
        elif perftype == 'AUC':
            assert(np.all(np.logical_or(testY == 0, testY==1))), 'AUC can be only calculated for binary measures'
            n = len(testY); num_pos = sum(testY==1); num_neg = n - num_pos
            Perf = 0.5 * np.ones(predmat.shape[1])
            if (num_pos > 0 & num_pos < n):
                ranks = rankdata(predmat,axis=0); Perf = ( sum( ranks[testY==1,:] ) - num_pos * (num_pos+1)/2) / ( num_pos * num_neg)
        elif perftype == 'Pearson':
            Perf = np.zeros(predmat.shape[1])
            for i in range(predmat.shape[1]):
                Perf[i] = pearsonr(testY,predmat[:,i])[0]
        elif perftype == 'Spearman':
            Perf = np.zeros(predmat.shape[1])
            for i in range(predmat.shape[1]):
                Perf[i] = spearmanr(testY,predmat[:,i])[0]
        return Perf

def TPLSinputchecker(datainput, name, dattype = None, maxval = None, minval = None, variation = 0, integercheck = 0):
    inputtype = type(datainput)
    assert (inputtype==int or inputtype==float or inputtype==np.ndarray), name + " should be numeric int or float" # numeric check
    assert (not np.isnan(datainput).any()), "NaN found in " + name # nan check
    assert (np.isfinite(datainput).all()), "Non finite value found in " + name # inf check

    if dattype is not None:
        if dattype == 'scalar':
            assert (inputtype==int or inputtype==float), name + " should be a scalar of type int or float"
        elif dattype == 'mat':
            assert(datainput.ndim == 2), name + " should have 2 dimensions as a matrix"
            n,v = datainput.shape
            assert (v > 2), name + " should have at least 3 columns"
            assert (n > 2), name + " should have at least 3 observations"
        elif dattype == 'colvec': # a column vector with 2 dimensions
            datainput = np.atleast_2d(datainput)
            n,v = datainput.shape
            assert (n == 1 or v == 1), name + " should be a vector" # it's okay if the input is a scalar which is a special case of column vector
            if v > 1 : # shouldn't be a row vector
                datainput = datainput.T
        elif dattype == 'rowvec': # a row vector with 1 dimensions
            if inputtype != int and inputtype != float:
                datainput = np.reshape(datainput,datainput.size)
        else:
            raise Exception("Unexpected input type checking requested")

    if maxval is not None:
        assert( np.all(datainput <= maxval) ), name + " should be less than or equal to " + maxval

    if minval is not None:
        assert( np.all(datainput >= minval) ), name + " should be greater than or equal to " + minval

    if variation == 1:
        assert( np.all(np.std(datainput)!=0) ), "There is no variation in " + name

    if integercheck == 1 and inputtype==float:
        assert(math.floor(datainput)==math.ceil(datainput)), name + " should be integer"
    elif integercheck == 1 and inputtype==np.ndarray:
        datainput = datainput.flatten()
        for i in range(datainput.size):
            assert(math.floor(datainput[i])==math.ceil(datainput[i])), name + " should be integer"

    return datainput