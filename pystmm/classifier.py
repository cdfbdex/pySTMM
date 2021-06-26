"""Module for classification function."""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class STMB(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Binary Classification using Support Tensor Machine with Linear Kernel only.
    """

    def __init__(self, C1=1.0, C2=1.0, maxIter=30, tolSTM=1e-4, penalty = 'l2', dual = True, tol=1e-4,loss = 'squared_hinge', maxIterSVM=100000):
        """Init."""
        # store params for cloning purpose
        self.C1 = C1
        self.C2 = C2
        self.dual = dual
        self.penalty = penalty
        self.tolSTM = tolSTM
        self.tol = tol
        self.loss = loss
        self.maxIter = maxIter
        self.maxIterSVM = maxIterSVM


    def fit(self, X, y):
        """Fit (estimates) the u and v.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : STMB instance
            The STMB instance.
        """
        self.classes_ = sorted([round(val) for val in np.unique(y)])
        self.dictclassestoval = dict(zip(self.classes_,[-1,+1]))
        self.dictvaltoclasses = dict(zip([-1,+1],self.classes_))
        y = np.array([self.dictclassestoval[round(val)] for val in y])

        #if self.n_jobs == 1:
        #STM: Support Tensor Machine
        n_channels = X.shape[1]
        n_trials = X.shape[0]
        n_samples = X.shape[2]
        u = np.ones(n_channels)
        v = np.ones(n_samples)
        #print(u.shape)

        totaliterations = self.maxIter
        for i in range(totaliterations):
          print('%d / %d'%(i+1,totaliterations))
          
          # First SVM on cols
          X_t = []
          for i in range(n_trials):
            x = X[i,:,:].T.dot(u)
            X_t.append(x)
          X_t = np.stack(X_t,axis=0)


          clf1 = LinearSVC(C=self.C1, dual=self.dual, penalty=self.penalty, loss=self.loss, tol=self.tol, max_iter=self.maxIterSVM)
          clf1.fit(X_t, y)
          coef_ = clf1.coef_
          intercept_ = clf1.intercept_
          n_iter_ = clf1.n_iter_


          v_next = coef_.copy()
          v_intercept = intercept_
          deltav = np.linalg.norm(v - v_next)
          v = v_next.copy()
          v_iter = n_iter_


          # Second SVM on rows
          X_c = []
          for i in range(n_trials):
            x = X[i,:,:].dot(v.T)[:,0]
            X_c.append(x)
          X_c = np.stack(X_c,axis=0)


          clf2 = LinearSVC(C=self.C2, dual=self.dual, penalty=self.penalty, loss=self.loss, tol=self.tol, max_iter=self.maxIterSVM)
          clf2.fit(X_c, y)
          coef_ = clf2.coef_
          intercept_ = clf2.intercept_
          n_iter_ = clf2.n_iter_

          u_next = coef_.copy()
          u_intercept = intercept_
          deltau = np.linalg.norm(u - u_next)
          u = u_next.copy()
          u_iter = n_iter_
          
          u = np.squeeze(u)
          self.deltau_ = deltau
          self.deltav_ = deltav

          if self.deltau_ < self.tolSTM and self.deltav_ < self.tolSTM:
            break
        self.u_opt_ = u_next.copy()
        self.v_opt_ = v_next.copy()
        self.intercept_ = u_intercept[0]

        X_pred = []
        for i in range(X.shape[0]):
          x = X[i,:,:]
          x_v = x.dot(self.v_opt_.T)
          x_u = float(self.u_opt_.dot(x_v))
          f = x_u + self.intercept_
          X_pred.append(f)
        X_pred = np.array(X_pred).reshape(X.shape[0],1)
        clf_log = LogisticRegression(random_state=0).fit(X_pred, y)
        self.LogisticRegression = clf_log
        
        return self

    def predict(self, X):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        pred = []
        for i in range(X.shape[0]):
          x = X[i,:,:]
          x_v = x.dot(self.v_opt_.T)
          x_u = float(self.u_opt_.dot(x_v))
          f = round(np.sign(x_u + self.intercept_))
          pred.append(self.dictvaltoclasses[f])      
        return np.array(pred)

    def decision_function(self, X):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_trials, rows, cols)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        pred = []
        for i in range(X.shape[0]):
          x = X[i,:,:]
          x_v = x.dot(self.v_opt_.T)
          x_u = float(self.u_opt_.dot(x_v))
          f = x_u + self.intercept_
          pred.append(f)      
        return np.array(pred)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        X_pred = []
        for i in range(X.shape[0]):
          x = X[i,:,:]
          x_v = x.dot(self.v_opt_.T)
          x_u = float(self.u_opt_.dot(x_v))
          f = x_u + self.intercept_
          X_pred.append(f)
        X_pred = np.array(X_pred).reshape(X.shape[0],1)
        y_proba = self.LogisticRegression.predict_proba(X_pred)
        return y_proba

class STMM(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Multiple Classification using Support Tensor Machine with Linear Kernel only.
    """

    def __init__(self, typemulticlassifier='ovr', C1=1.0, C2=1.0, maxIter=30, tolSTM=1e-4, penalty = 'l2', dual = True, tol=1e-4,loss = 'squared_hinge', maxIterSVM=100000):
        self.typemulticlassifier = None
        print('maxIter ',maxIter)
        print(typemulticlassifier, C1, C2, maxIter, tolSTM, penalty, dual, tol, loss, maxIterSVM)
        if typemulticlassifier=='ovr':
            self.clf = OneVsRestClassifier(STMB(C1=C1, C2=C2, maxIter=maxIter, tolSTM=tolSTM, penalty = penalty, dual = dual, tol=tol,loss = loss, maxIterSVM=maxIterSVM))
            self.typemulticlassifier = typemulticlassifier
            self.C1 = C1
            self.C2 = C2
            self.maxIter = maxIter
            self.tolSTM = tolSTM
            self.penalty = penalty
            self.dual = dual
            self.tol = tol
            self.loss = loss
            self.maxIterSVM = maxIterSVM
        elif typemulticlassifier=='ovo':
            self.clf = OneVsOneClassifier(STMB(C1=C1, C2=C2, maxIter=maxIter, tolSTM=tolSTM, penalty = penalty, dual = dual, tol=tol,loss = loss, maxIterSVM=maxIterSVM))
            self.typemulticlassifier = typemulticlassifier
            self.C1 = C1
            self.C2 = C2
            self.maxIter = maxIter
            self.tolSTM = tolSTM
            self.penalty = penalty
            self.dual = dual
            self.tol = tol
            self.loss = loss
            self.maxIterSVM = maxIterSVM
        else:
            print('Set a valid -typemulticlassifier- (ovr or ovo).')

    def fit(self, X, y, sample_weight=None):
        self.clf.fit(X, y)
        return self  

    def predict(self, X):
        return self.clf.predict(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.clf.fit(X, y)
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)