"""Module for classification function."""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from hottbox.core import Tensor
import copy
import time
from scipy.spatial.distance import pdist, cdist, squareform


def contractor(x, w, modes):
    """
    Parameters
    ----------
    x: Tensor object
    w: weights for STM to be contracted against
    modes: modes for STM to be contracted against
    Returns
    -------
    x_vec = contracted tensor along all modes except for one
    """

    temp = x.copy()
    for w, mode in zip(w, modes):
        temp.mode_n_product(np.expand_dims(w, axis=0), mode, inplace=True)
    x_vec = np.expand_dims(temp.data.squeeze(), axis=0)

    return x_vec

class LSSTM:
    def __init__(self, C=10, kernel='linear', sig2=1, max_iter=100):

        self.order = None
        self.shape = None
        self.C = C
        self.max_iter = max_iter
        self.model = {'Weights': None,
                      'Bias': 0,
                      'nIter': 0
                      }

        self.eta_history = []
        self.b_history = []
        self.orig_labels = None
        self.kernel = kernel
        self.sig2 = sig2



    def fit(self, X_train, labels):
        """
        Parameters
        ----------
        X_train: list[Tensor],  list of length M of Tensor objects, all of the same order and size
        labels: list of length M of labels +1, -1
        Returns
        -------
        """
        self.order = X_train[0].order
        self.shape = X_train[0].shape
        self._assert_data(X_train, labels=labels)

        self.orig_labels = list(set(labels))
        labels = [1 if x == self.orig_labels[0] else -1 for x in labels]

        w_n = self._initialize_weights(X_train[0].shape)

        for i in range(self.max_iter):
           # w_n_old = copy.deepcopy(weights)
            for n in range(self.order):
                #Always seems to be better if the weights are updated on the fly, rather than altogether at the
                #end of each iteration
                # eta = self._calc_eta(w_n_old, n)
                # X_m = self._calc_Xm(X_train, w_n_old, n)
                eta = self._calc_eta(w_n, n)
                X_m = self._calc_Xm(X_train, w_n, n)
                self.eta_history.append(eta)

                w, b = self._compute_weights(X_m, labels, eta, self.C, kernel=self.kernel, sig2=self.sig2)
                w = w / np.linalg.norm(w)
                w_n[n] = w


            self._update_model(w_n, b, i)
            if self._converged(): break



    def predict(self, X_test):
        """
        Parameters
        ----------
        X_test: list[Tensor]
        Returns
        -------
        y_pred: list of predicted laels
        """
        #if singleton
        if not isinstance(X_test, list):
            X_test = [X_test]

        self._assert_data(X_test)

        w_n = self.model['Weights']
        b = self.model['Bias']
        y_pred = []
        dec_values = []
        for xtest in X_test:
            temp = xtest.copy()
            for n, w in enumerate(w_n):
                temp.mode_n_product(np.expand_dims(w, axis=0), mode=n, inplace=True)
            dec_values.append(temp.data.squeeze() + b)
            y_pred.append(np.sign(temp.data.squeeze() + b))

        y_pred = [self.orig_labels[0] if x == 1 else self.orig_labels[1] for x in y_pred]
        return y_pred, dec_values



    def _assert_data(self, X_data, labels=None):
        """
        Parameters
        ----------
        X_data: list[Tensor]
        Returns
        -------
        None, just checks if all tensors have same order and dimensions, and if labels are binary
        """
        order = self.order
        shape = self.shape
        for tensor in X_data[1:]:
            assert tensor.order == order, "Tensors must all be of the same order"
            assert tensor.shape == shape, "Tensors must all have modes of equal dimensions"
            order = tensor.order
            shape = tensor.shape

        if labels is not None:
            assert len(set(labels)) == 2, "LSSTM is a binary classifier, more than two labels were passed"


    def _initialize_weights(self, shape):
        """
        Parameters
        ----------
        shape: tuple, of tensor dimensions
        Returns
        -------
        w_n: list, the initialized weights
        """
        w_n = []
        for dim in shape:
            w_n.append(np.random.randn(dim))
        return w_n


    def _calc_eta(self, w_n, n):
        """
        Parameters
        ----------
        w_n: list, of LS-STM weights
        n: int, the one to leave out
        Returns
        -------
        eta: int, parameter to be used in LS-STM optimization problem
        """

        w_n_new = [w for i, w in enumerate(w_n) if i != n]
        eta = 1
        for w in w_n_new:
            eta *= (np.linalg.norm(w)**2)
        return eta


    def _calc_Xm(self, X_data, w_n, n):
        """
        Parameters
        ----------
        X_data: list[Tensor], all the data as list of tensor objects
        w_n: list, the weights treated as constants
        n: int, the mode we're looking at
        Returns
        -------
        X_m: np.ndarray of size M x mode(n), to be passed to the LS-SVM solver
        """
        order = X_data[0].order
        w_n_new = [w for i, w in enumerate(w_n) if i != n]
        modes = [i for i in range(order) if i != n]

        result = list(map(lambda x: contractor(x, w_n_new, modes), X_data))
        X_m = np.array(result).squeeze()

        return X_m


    def _compute_weights(self, X_m, labels, eta, C, kernel, sig2):
        """
        Parameters
        ----------
        X_m: np.ndarray,  Matrix of contracted tensors along all weights except the current n
        labels: int, the labels of hte training data
        eta: int, Parameter to be used in the algo
        C: cost
        kernel: which kernel to use (linear, rbf, etc)
        Returns
        -------
        w: list, Weights for mode n
        b: int, Bias
        """
        M = X_m.shape[0]
        if kernel=='linear':
            alphas, b = self._ls_optimizer(X_m, labels, eta, C, kernel=kernel, sig2=sig2)
            w = np.sum(alphas * X_m, axis=0)
        elif kernel=='RBF':
            y = np.zeros(M)
            alphas, b = self._ls_optimizer(X_m, labels, eta, C, kernel=kernel, sig2=sig2)
            for i in range(M):
                x_star = X_m[[i]]
                X_tmp = np.delete(X_m, i, axis=0)
                l_tmp = np.delete(labels, i, axis=0)
                alpha_tmp = np.delete(alphas, i, axis=0)[:,0]
                rbf_vector = np.exp(-np.square(cdist(X_tmp, x_star)[:,0])/(2*sig2))
                y[i] = np.sum(alpha_tmp*l_tmp*rbf_vector)
            w = np.dot(np.linalg.pinv(X_m), y-b)


        return w, b



    def _ls_optimizer(self, X_m, labels, eta, C, kernel, sig2):
        """
        Parameters
        ----------
        X_m: np.ndarray,  Matrix of contracted tensors along all weights except the current n
        labels: int, the labels of hte training data
        eta: int, Parameter to be used in the algo
        C: Cost
        kernel: which kernel to use (linear, rbf, etc)
        Returns
        -------
        alphas: the alphas computed from the Lagrangian. The first alpha is the b, the bias parameter
        b = the bias parameter
        """

        M = X_m.shape[0]
        gamma = C / eta
        y_train = np.expand_dims(np.array(labels), axis=1)

        #For now, use no kernel
        if kernel=='linear':
            Omega = np.dot(X_m, X_m.transpose())
        elif kernel=='RBF':
            Omega = np.exp(-np.square(squareform(pdist(X_m))) / (2*sig2))

        left_column = np.expand_dims(np.append(np.array([0]), np.ones(M)), axis=1)
        right_block = np.append(np.expand_dims(np.ones(M), axis=0),  Omega + (1/gamma) * np.eye(M), axis=0)
        params = np.append( left_column, right_block, axis=1)

        RHS = np.append(np.array([[0]]), y_train, axis=0)

        alphas = np.dot(np.linalg.inv(params), RHS)

        b = alphas[0][0]
        alphas = alphas[1:, :]

        return alphas, b



    def _update_model(self, w, b, nIter):
        """
        Parameters
        ----------
        w: list, estimated weights
        b: int,  estimated bias
        nIter: int, current iteration number
        Returns
        -------
        None
        """
        self.model['Weights'] = w
        self.model['Bias'] = b
        self.model['nIter'] = nIter



    def _converged(self):
        """
        Parameters
        ----------
        w_n_old: list, previous computed weights
        Returns
        -------
        Boolean
        """

        self.b_history.append(self.model['Bias'])

        if len(self.b_history) > 10:
            err1 = np.diff(np.array(self.eta_history[-11:]))
            err2 = np.diff(np.array(self.b_history[-11:]))

            if np.all(np.abs(err1) < 1e-8) and np.all(np.abs(err2) < 1e-8):
                return True

        return False


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