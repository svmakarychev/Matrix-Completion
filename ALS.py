import numpy as np
import scipy.sparse as sparse
from scipy.linalg import solve
#from scipy.sparse.linalg import norm
from copy import deepcopy
from collections import defaultdict
from time import time
from scipy.sparse import linalg

class ALS(object):
    """
    Realisation of ALS algorithm.
    Provides matrix completion procedure using Alternating Least Squares.
    
    For desription of the algorithm and notation see 
    http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf
    """
    def __init__(self, rank=10, max_iter = 10, tol=1e-5, reg_coef=1e-5):
        """
        ALS instance takes the following parameters for initialization:
        
        Args:
            rank (int), rank of approximation
            max_iter (int): number of iterations of algorithm
            tol (float): early stop constraint for the algorithm
            reg_koef (float): regularization constant
        """
                
        self._rank = rank
        self._max_iter = max_iter
        self._tol = tol
        self._reg_coef = reg_coef    
     
    def Initialization(self, X):
        """
        Random initialization of the matrix A and B in the form:
        A = UD, B = VD,  
        where U, V - orthogonal matrices, D - diagonal of ones.
        """
        
        np.random.seed(self._random_state)
        
        self._X = X
        self._m = self._X.shape[0]
        self._n = self._X.shape[1]

        self._Omega = self._X.nonzero()
        
        Random = np.random.uniform(0, 1, (self._m, self._rank))
        Qu, Ru = np.linalg.qr(Random)
        self._A = Qu.T
        
        Random = np.random.uniform(0, 1, (self._n, self._rank))
        Qv, Rv = np.linalg.qr(Random)
        self._B = Qv.T
        
        pass

    
    def projection(self, Y1, Y2):
        """
        Make projection on the space on known entries. Assume we have the following representation X = Y1Y2.T
        Args:
            Y1 (np.array): first matrix in the representation X = Y1Y2.T
            Y2 (np.array)L second matrix in the representation X = Y1Y2.T
        Returns:
            proj (np.array): Projection on the known entries.
        """
        rows = self._Omega[0]
        cols = self._Omega[1]
        
        nnz = rows.shape[0]
        data = np.zeros(rows.shape[0])
        
        for it in range(0, nnz):
            data[it] = np.dot(Y1[rows[it]].T, Y2[cols[it]])
        
        proj = sparse.csr_matrix((data, (rows, cols)), shape=self._X.shape)
        
        return proj
    
    def fit(self, X, trace=False, debug_mode=False, random_state=123):
        """
        This method runs ALS with given matrix.
        
        ratings - (m, n) matrix, matrix with missed values to be complited.
        """
        self._random_state = random_state
        self.Initialization(X)         
        
        history = defaultdict(list) if trace else None
        
        A_old = self._A
        B_old = self._B
        AtB_old = A_old.T.dot(B_old)
                
        iters=0
        
        if trace:
            start_time = time()
            f_old = linalg.norm(self._X - self.projection(self._A.T, self._B.T), ord='fro')**2
            f_old = f_old + self._reg_coef * (np.linalg.norm(self._A)**2 + np.linalg.norm(self._B)**2)
        
        
        while True:
        
            if(iters >= self._max_iter):
                self._message = "Iteration exceeded. Did not converge"
                break
            
            iters+=1
            
            for j in range(self._m):
                #take row in matrix of observations with which we will work
                row = self._X.getrow(j).toarray().ravel()

                #find arguments of elemnts in row which were not missed
                args = np.argwhere(row != 0).T[0]

                #create matrices to save temporary results
                summation_inv = np.zeros((self._rank, self._rank))
                summation = np.zeros((self._rank, 1))

                #for every non-nan element in row we take corresponding column of Y and make manipulations
                for arg in args:
                    summation_inv = summation_inv + (self._B[:, arg].reshape(-1,1)).dot(self._B[:, arg].reshape(1, -1))
                    summation = summation + row[arg] * self._B[:, arg].reshape(-1,1)

                #update the corresponding column of X
                new_A = solve(summation_inv + self._reg_coef * np.eye(self._rank), summation)
                self._A[:,j] = new_A.reshape(-1,)
             

            #repeat everything for matrix Y
            for j in range(self._n):
                #take column in matrix of observations with which we will work
                column = self._X.getcol(j).toarray().ravel()

                #find arguments of elemnts in column which were not missed
                args = np.argwhere(column != 0).T[0]

                #create matrices to save temporary results
                summation_inv = np.zeros((self._rank, self._rank))
                summation = np.zeros((self._rank, 1))

                #for every non-nan element in row we take corresponding column of Y and make manipulations
                for arg in args:
                    summation_inv = summation_inv + (self._A[:, arg].reshape(-1,1)).dot(self._A[:, arg].reshape(1, -1))
                    summation = summation + column[arg] * self._A[:, arg].reshape(-1,1)

                #update the corresponding column of Y
                new_B = solve(summation_inv + self._reg_coef * np.eye(self._rank), summation)
                self._B[:,j] = new_B.reshape(-1)

            AtB = self._A.T.dot(self._B)
            
            diff_norm = np.linalg.norm(AtB_old - AtB) ** 2 
            
            if trace:
                f = linalg.norm(self._X - self.projection(self._A.T, self._B.T), ord='fro')**2
                history['residuals'].append(f)
                
                f = f + self._reg_coef * (np.linalg.norm(self._A)**2 + np.linalg.norm(self._B)**2)
                
                history['func'].append(f)
                
                history['diff_norm'].append(diff_norm)
                
                cur_time = time()
                history['time'].append(time() - start_time)
                
                history['rel_obj'].append((f - f_old)/f_old)
                f_old = f
                
            if (diff_norm < self._tol):
                self._message = "Accuracy achieved. Converged."
                break
            
            else:
                AtB_old = AtB
            
            if debug_mode:
                print("Number of iteration: ", iters, " ", "Diff_norm: ", diff_norm)
            
        
        self._history = history
        self._solution = AtB    
            
        pass
        
    
    def get_factors(self):
        """
        Returns A and B - factorization of matrix, used in fit method
        """
        return self._A, self._B
        
    def predict(self):
        """
        Returns the product of factorization matrices B and B: AB^T
        """
        return (self._A.T).dot(self._B)
    
    def get_residuals(self):
        """
        Return residuals Proj(X - AB^T) on each iteration
        Use only if trace = True
        """
        
        try:
            ans = self._history['residuals']
        
        except TypeError:
            ans = "Error, there is no attribute with such name. Check whether trace == True." 
        
        return ans
    
    def get_time(self):
        """
        Return time per each iteration
        Use only if trace = True
        """
        
        try:
            ans = self._history['time']
        
        except TypeError:
            ans = "Error, there is no attribute with such name. Check whether trace == True."
        
        return ans
    
    def get_func(self):
        """
        Return value of minimising functional per each iteration
        Use only if trace = True
        """
        
        try:
            ans = self._history['func']
        
        except TypeError:
            ans = "Error, there is no attribute with such name. Check whether trace == True."
        
        return ans
    
    def get_rel_obj(self):
        """
        Get relevance objective per each iteration
        """
        
        try:
            ans = self._history['rel_obj']
        
        except TypeError:
            ans = "Error, there is no attribute with such name. Check whether trace == True."
        
        return ans
    
    def get_diff_norm(self):
        """
        Return squared Frobenius norm of difference of approximations AB^T
        on the previous iteration and current iteration
        Use only if trace = True
        """
        
        try:
            ans = self._history['diff_norm']
        
        except TypeError:
            ans = "Error, there is no attribute with such name. Check whether trace == True."
        
        return ans
    
    def get_message(self):
        """
        Return message of the result of fit method:
        Converged or did not converge.
        """
        return self._message