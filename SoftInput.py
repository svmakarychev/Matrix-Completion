import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sparse

class SoftInput():
    """Realization of Soft-Input algorithm for matrix completion.
       For desription of the algorithm and notation see https://web.stanford.edu/%7Ehastie/Papers/mazumder10a.pdf

    Attributes:
    bias: array with Frobenius distances between X and a projection of current approximation at each step 

    """

    def __init__(self, X):
        """Initialization of a class issue

        Args:
        X: sparse matrix to complete

        """
        self.X = X
        self.nonzero = self.X.nonzero()
        
    def fit(self, lambdas, maxiter=np.inf, tol=1e-6, start_rank=100):
        """Complete matrix X, return series of low-rank approximations

        Args:
        lambdas: regularization parameters for fitting
        maxiter: maximem number of made iterations
        tol: tolerance, if Frobenius norm of the difference of two approximations is less than tol,
             algorithm is treated as converged
        start_rank: rank of the initial approximation

        Returns:
        lambdas: descendingly sorted lambdas from Args
        U_approx, Vt_approx: U_approx[i].dot(Vt_approx[i]) is an i-th approximation for X

        """
        self.rank = start_rank
        self.tol = tol
        self.lambdas = np.sort(lambdas)[::-1]
        self.Z_proj = sparse.csr_matrix(self.X.shape)
        self.S = np.zeros((self.rank,))
        self.U = np.zeros((self.X.shape[0], self.rank))
        self.Vt = np.zeros((self.rank, self.X.shape[1]))
        
        self.U_approx = []
        self.Vt_approx = []
        
        self.bias = np.array([])
        
        num_iter = 0

        for l in self.lambdas:
            while (num_iter < maxiter):
                self.update_Z_proj()
                self.bias = np.append(self.bias, spla.norm(self.Z_proj - self.X, ord='fro'))
                
                lin_op = spla.LinearOperator(self.X.shape,
                                             matvec=self.matvec,
                                             rmatvec=self.rmatvec)
                U, S, Vt = spla.svds(lin_op, self.rank)
                
                S -= l
                S[S < 0] = 0
                
                if (num_iter > 0):
                    rel_err = np.linalg.norm(S - self.S) ** 2 / np.linalg.norm(self.S) ** 2
                else:
                    rel_err = 2 * self.tol
                
                S = S[S > 0]
                self.rank = S.shape[0]
                
                U = U[:, :self.rank]
                Vt = Vt[:self.rank, :]
                
                self.U = U
                self.Vt = Vt
                self.S = S
                
                self.Vt = np.diag(self.S).dot(self.Vt)
                
                if (rel_err < self.tol):
                    break
                
                num_iter += 1
                
            self.U_approx.append(self.U)
            self.Vt_approx.append(self.Vt)
        
        return self.lambdas, self.U_approx, self.Vt_approx
    
    def update_Z_proj(self):
        """Set self.Z_proj to be a sparse matrix containing a projection of a current approximation
           on the set of known values

        """
        proj_data = np.empty(self.nonzero[0].size)
        for i in range(self.nonzero[0].size):
            proj_data[i] = self.U[self.nonzero[0][i], :].dot(self.Vt[:, self.nonzero[1][i]])
        self.Z_proj = sparse.csr_matrix((proj_data, self.nonzero), self.X.shape)
            

    def matvec(self, vec):
        """MATVEC for P_{\Omega}(X) - P_{\Omega}(Z) + Z

        """
        res = self.X.dot(vec) - self.Z_proj.dot(vec)
        res += self.U.dot(self.Vt.dot(vec))
        return res
    
    def rmatvec(self, vec):
        """MATVEC for P_{\Omega}(X) - P_{\Omega}(Z) + Z

        """
        res = self.X.T.dot(vec) - self.Z_proj.T.dot(vec)
        res += self.Vt.T.dot(self.U.T.dot(vec))
        return res

