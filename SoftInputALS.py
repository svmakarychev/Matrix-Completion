import numpy as np
import scipy.sparse as sparse
import copy
#import scipy.sparse as scp
from scipy.sparse import linalg
from collections import defaultdict
from time import time

class SoftInputALS():
    """Realization of Rank-Restricted Efficient Maximum-Margin Matrix Factorization: SoftInput - ALS:
        
       Solving the following problem: 
       
       minimize_{A, B}\|P_{\Omega}(X-AB^T)\|_F^2 + \lambda(\|A\|_F^2 + \|B\|_F^2)
       
       For desription of the algorithm and notation see http://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf

    Attributes:
    residuals: array with Frobenius distances between X and a projection of current approximation at each step
    rel_objective: array with relative objective values
    """

    def __init__(self, X, rank, maxiter, tol, regcoef):
        """
        Initialization method.
        Args:
            X (np.array): given matrix with some unknown entries
            rank (int): Rank of approximation matrix
            maxiter (int): Number of iterations for early stop
            accuracy (float): Accuracy of solution for early stop
            regcoef (float): Regularization coefficient
        """
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
        self.regcoef = regcoef
        
        self.X = X
        self.m = X.shape[0]
        self.n = X.shape[1]

        self.Omega = X.nonzero()

    def projection(self, Y1, Y2):
        """
        Make projection on the space on known entries. Assume we have the following representation X = Y1Y2.T
        Args:
            Y1 (np.array): first matrix in the representation X = Y1Y2.T
            Y2 (np.array): second matrix in the representation X = Y1Y2.T
        Returns:
            proj (np.array): Projection on the known entries.
        """
        rows = self.Omega[0]
        cols = self.Omega[1]
        
        nnz = rows.shape[0]
        data = np.zeros(rows.shape[0])

        for it in range(0, nnz):
            data[it] = np.dot(Y1[rows[it]].T, Y2[cols[it]])
        
        proj = sparse.csr_matrix((data, (rows, cols)), shape=self.X.shape)
        
        return proj
    
    
    def Initialization(self):
        """
        Random initialization of the matrix A and B in the form:
        A = UD, B = VD,  
        where U, V - orthogonal matrices, D - diagonal of ones.
        """

        m = self.m
        n = self.n
        r = self.rank
        
        Random = np.random.uniform(0, 1, (m, r))
        Qu, Ru = np.linalg.qr(Random)
        self.U = Qu
        
        Random = np.random.uniform(0, 1, (n, r))
        Qv, Rv = np.linalg.qr(Random)
        self.V = Qv
        
        self.D = np.identity(r)
        
        self.A = self.U.dot(self.D)
        self.B = self.V.dot(self.D)
        
        pass

    
   
    def fit(self, trace = False):
        """
        Main algorithm for solving optimization problem.
        Set self.approx as approximation matrix
        Set self.reiduals as residuals of objective function on each iteration.
        Set self.solution as the solution of the problem.
        Set self.message as the result of fitting: Converge or did not converge
        Set self.times as time spending from the start of the fitting on each iteration
        Set self.rel_objective as the objective value per each iteration
        """
        
        history = defaultdict(list) if trace else None
        
        residuals = []
        
        self.Initialization()
        
        A_old = self.A
        B_old = self.B
        
        ABt_old = A_old.dot(B_old.T)
                
        iters=0
        if trace:
            start_time = time.time()
        
        while True:
            
            if(iters > self.maxiter):
                self.message = "Iteration exceeded. Did not converge"
                break
            iters+=1
            
            Sparse_part = self.X - self.projection(self.A, self.B)
            Low_rank_part = self.A.dot(self.B.T)
            
            X_star = Sparse_part + Low_rank_part
            
            Inv = np.linalg.inv(self.B.T.dot(self.B) + self.regcoef * np.identity(self.rank))
            self.A = np.asarray(X_star.dot(self.B).dot(Inv))

            Sparse_part = self.X - self.projection(self.A, self.B)
            Low_rank_part = self.A.dot(self.B.T)
            
            X_star = Sparse_part + Low_rank_part
            
            Inv = np.linalg.inv(self.A.T.dot(self.A) + self.regcoef * np.identity(self.rank))
            self.B = np.asarray(X_star.T.dot(self.A).dot(Inv))
        
            ABt = self.A.dot(self.B.T)
            
            if trace:
                f = linalg.norm(self.X - self.projection(self.A, self.B), ord='fro')**2
                f = f + self.regcoef * (np.linalg.norm(self.A)**2 + np.linalg.norm(self.B)**2)
                history['func'].append(f)
                history['ABt'].append(ABt)
                history['diff_norm'].append(np.linalg.norm(ABt_old - ABt) ** 2)
                cur_time = time.time()
                history['time'].append(time.time() - start_time)
            
            
            if (np.linalg.norm(ABt_old - ABt) ** 2 < self.tol):
                self.message = "Accuracy achieved. Converged."
                break
            
            else:
                ABt_old = ABt
            
            residuals.append(linalg.norm(self.X - self.projection(self.A, self.B), ord='fro')**2)
        
        self.residuals = residuals
        self.history = history
        self.solution = ABt    
            
        pass