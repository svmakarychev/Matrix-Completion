import numpy as np
import scipy.sparse as sparse
import copy
from scipy.sparse import linalg
from collections import defaultdict
from time import time
from copy import deepcopy

class FastALS():
    """Realization of Rank-Restricted Efficient Maximum-Margin Matrix Factorization: SoftInput - ALS:
        
       Solving the following problem: 
       
       minimize_{A, B}\|P_{\Omega}(X-AB^T)\|_F^2 + \lambda(\|A\|_F^2 + \|B\|_F^2)
       
       For desription of the algorithm and notation see http://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf

    """

    def __init__(self, rank=10, max_iter=10, tol=1e-5, reg_coef=1e-5):
        """
        Initialization method.
        Args:
            rank (int): Rank of approximation matrix
            maxiter (int): Number of iterations for early stop
            accuracy (float): Accuracy of solution for early stop
            reg_coef (float): Regularization coefficient
        """
        self._rank = rank
        self._max_iter = max_iter
        self._tol = tol
        self._reg_coef = reg_coef

    def projection(self, Y1, Y2):
        """
        Make projection on the space on known entries. Assume we have the following representation X = Y1Y2.T
        Args:
            Y1 (np.array): first matrix in the representation X = Y1Y2.T
            Y2 (np.array): second matrix in the representation X = Y1Y2.T
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
        self._U = Qu
        
        Random = np.random.uniform(0, 1, (self._n, self._rank))
        Qv, Rv = np.linalg.qr(Random)
        self._V = Qv
        
        self._D = np.identity(self._rank)
        
        self._A = self._U.dot(self._D)
        self._B = self._V.dot(self._D)
        
        pass

   
    def fit(self, X, trace = False, debug_mode = False, random_state = 123):
        """
        
        This is the main algorithm for solving optimization problem.
        
        Args:
            X (csr_matrix): matrix to complete
            trace (bool): Whether to safe history
            debug_mode (bool): If True, print diff_norm ||ABt_old - ABt|| ** 2 and number of iteration
        
        if trace id True, than the following information will be collected and safed in self.history dict:

            history['func']: value of the function per each iteration
            history['diff_norm']: value of difference ||ABt_old - ABt|| ** 2 per each iteration
            history['time']: time spending from the start of the fitting on each iteration
            history['rel_obj']: Value of relevance objective per each iteration
            history['residuals']: ||self._X - self.projection(self._A, self._B)||^2
            
        Also safes the following attributes:
            self.solution: Approximation of the X at the end of fitting precedure
            self.message: Message of convergence/not convergence of algorithm
        
        """
        
        self._random_state = random_state
        self.Initialization(X)
        
        history = defaultdict(list) if trace else None
                
        A_old = self._A
        B_old = self._B
        
        ABt_old = A_old.dot(B_old.T)
                
        iters=0
        if trace:
            start_time = time()
            f_old = linalg.norm(self._X - self.projection(self._A, self._B), ord='fro')**2
            f_old = f_old + self._reg_coef * (np.linalg.norm(self._A)**2 + np.linalg.norm(self._B)**2)
        
        while True:
            
            if(iters >= self._max_iter):
                self._message = "Iteration exceeded. Did not converge"
                break
            
            iters+=1
            
            Sparse_part = self._X - self.projection(self._A, self._B)
            Low_rank_part = self._A.dot(self._B.T)
            
            X_star = Sparse_part + Low_rank_part
            
            Inv = np.linalg.inv(self._B.T.dot(self._B) + self._reg_coef * np.identity(self._rank))
            self._A = np.asarray(X_star.dot(self._B).dot(Inv))

            Sparse_part = self._X - self.projection(self._A, self._B)
            Low_rank_part = self._A.dot(self._B.T)
            
            X_star = Sparse_part + Low_rank_part
            
            Inv = np.linalg.inv(self._A.T.dot(self._A) + self._reg_coef * np.identity(self._rank))
            self._B = np.asarray(X_star.T.dot(self._A).dot(Inv))
        
            ABt = self._A.dot(self._B.T)
            
            diff_norm = np.linalg.norm(ABt_old - ABt) ** 2 
            
            if trace:
                f = linalg.norm(self._X - self.projection(self._A, self._B), ord='fro')**2
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
                ABt_old = ABt
            
            if debug_mode:
                print("Number of iteration: ", iters, " ", "Diff_norm: ", diff_norm)
            
        
        self._history = history
        self._solution = ABt    
            
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
        return (self._A).dot(self._B.T)
    
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