import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr
import matplotlib.pyplot as plt

class evalTuningParam:
    
    def __init__(self, TPLScvmdl, type, X, Y, compvec, threshvec, subfold = 0): 
        """ Evaluating tuning parameters from cross-validated TPLS model
            Arguments:
            TPLScvmdl -- Cross-validated TPLS model created by TPLS_cv
            type -- Cross-validation performance measure. One of 'AUC', 'pearson', 'spearman'
            X -- The same X used to train TPLScvmdl
            Y -- The same Y used to train TPLScvmdl
            compvec -- vector of number of components you want to evaluate for performance
            threshvec -- vector of threshold values you want to evaluate for performance
            subfold -- vector of additional division within the testing fold to calculate performance. For example, run number can be provided to calculate run-level performance and average them at the subject level
        """
        # assert (), 'performance metric must be one of ''Pearson'',''Spearman'',or ''AUC'''
        if subfold == 0:
            subfold = TPLScvmdl.testfold
        
        # Perform CV prediction and performance measurement
        threshvec = np.sort(threshvec); compvec = np.sort(compvec) # sorted from low to high
        perfmat = np.empty((len(compvec),len(threshvec),TPLScvmdl.numfold)); perfmat.fill(np.nan)

        for i in range(TPLScvmdl.numfold):
            print('Fold #'+str(i+1))
            testCVfold = (TPLScvmdl.testfold == i)
            Ytest = Y[testCVfold]
            testsubfold = subfold[testCVfold]
            uniqtestsubfold = np.unique(testsubfold)
            for j in range(len(threshvec)):
                predmat = TPLScvmdl.cvMdls[i].predict(compvec,threshvec[j],X[testCVfold.flatten(),:])
                smallperfmat = np.empty((len(compvec),len(uniqtestsubfold)))
                for k in range(len(uniqtestsubfold)):
                    subfoldsel = testsubfold == uniqtestsubfold[k]
                    smallperfmat[:,k] = self.util_perfmetric(predmat[subfoldsel.flatten(),:],Ytest[subfoldsel],type)
                perfmat[:,j,i] = np.nanmean(smallperfmat, axis=1)
        
        # prepare output object
        self.type = type; self.threshval = threshvec; self.compval = compvec; self.perfmat = perfmat

        # find the point of maximum CV performance
        outlist = self.findBestPerf(perfmat)
        self.perf_best = outlist[0]; self.compval_best = compvec[outlist[1]]; self.threshval_best = threshvec[outlist[2]]
        self.perf_1se = outlist[3]; self.compval_1se = compvec[outlist[4]]; self.threshval_1se = threshvec[outlist[5]]

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
        plt.show()
    
    @staticmethod
    def findBestPerf(perfmat):
        avgperfmat = np.nanmean(perfmat, axis=2); perf_best = np.nanmax(avgperfmat)
        row_best,col_best = np.where(avgperfmat==perf_best)
        row_best = row_best[0]; col_best = col_best[0]
        standardError = np.nanstd(perfmat[row_best,col_best])/perfmat.shape[2] # finding the standard error of the best point
        candidates = avgperfmat[:,np.arange(col_best+1)] > (perf_best-standardError) # finding points whose mtric is higher than perf_max minus 1 SE
        col_1se,row_1se = np.where(candidates.T)
        row_1se = row_1se[0]; col_1se = col_1se[0]
        perf_1se = avgperfmat[row_1se,col_1se]
        return [perf_best,row_best,col_best,perf_1se,row_1se,col_1se]

    @staticmethod
    def util_perfmetric(predmat,testY,type):
        if type == 'AUC':
            n = len(testY); num_pos = sum(testY==1); num_neg = n - num_pos
            if (num_pos > 0 & num_pos < n):
                ranks = rankdata(predmat,axis=0); Perf = ( sum( ranks[testY==1,:] ) - num_pos * (num_pos+1)/2) / ( num_pos * num_neg)
        else:
            Perf = np.zeros(predmat.shape[1])
            if type == 'pearson':
                for i in range(predmat.shape[1]):
                    Perf[i] = pearsonr(testY,predmat[:,i])[0]
            else:
                for i in range(predmat.shape[1]):
                    Perf[i] = spearmanr(testY,predmat[:,i])[0]
            Perf[np.isnan(Perf)]=0
        return Perf