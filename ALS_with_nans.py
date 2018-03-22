import numpy as np
from scipy import sparse
from scipy.linalg import solve


class ALS(object):
    """
    ALS
    
    Provides matrix completion procedure using Alternating Least Squares
    """
    def __init__(self, k=10, lambda_ = 10, max_iter=10, tol=1e-5, missed_val = 'NaN', print_iter=False):
        """
        ALS instance takes the following parameters for initialization:
        k - float, rank of approximation
        
        lambda_ - float, regularization constant
        
        max_iter - int, number of iterations of algorithm
        
        tol - float, early stop constraint for the algorithm
        
        print_iter - True of False, indicates whether algorithm prints number of current iteration
        """
        self._k = k
        self._lambda = lambda_
        self._max_iter = max_iter
        self._missed_val = missed_val
        self._tol = tol
        self._print_iter = print_iter
        self._errors = []
    
    def fit(self, ratings):
        """
        This method runs ALS with given matrix.
        
        ratings - (m, n) matrix, matrix with missed values to be complited.
        """
        #initial assigning of factors X, Y
        X, Y = np.abs(np.random.rand(self._k, ratings.shape[0])), np.abs(np.random.rand(self._k, ratings.shape[1]))
        for i in range(self._max_iter):
            column_diff_norm = 0
            if self._print_iter:
                print(i)
            
            for j in range(ratings.shape[0]):
                #take row in matrix of observations with which we will work
                row = ratings.getrow(j).toarray().ravel()

                #find arguments of elemnts in row which were not missed
                args = []
                if self._missed_val == 'NaN':
                    args = np.argwhere(~np.isnan(row)).T[0]
                else:
                    args = np.argwhere(row != self._missed_val).T[0]

                #create matrices to save temporary results
                summation_inv = np.zeros((self._k, self._k))
                summation = np.zeros((self._k, 1))

                #for every non-nan element in row we take corresponding column of Y and make manipulations
                for arg in args:
                    summation_inv = summation_inv + (Y[:, arg].reshape(-1,1)).dot(Y[:, arg].reshape(1, -1))
                    summation = summation + row[arg] * Y[:, arg].reshape(-1,1)

                #update the corresponding column of X
                new_X = solve(summation_inv + self._lambda * np.eye(self._k), summation)
                column_diff_norm += np.linalg.norm(X[:,j] - new_X) / np.linalg.norm(X[:,j])
                X[:,j] = new_X.reshape(-1,)

            #repeat everything for matrix Y
            for j in range(ratings.shape[1]):
                #take column in matrix of observations with which we will work
                column = ratings.getcol(j).toarray().ravel()

                #find arguments of elemnts in column which were not missed
                args = []
                if self._missed_val == 'NaN':
                    args = np.argwhere(~np.isnan(column)).T[0]
                else:
                    args = np.argwhere(column != self._missed_val).T[0]

                #create matrices to save temporary results
                summation_inv = np.zeros((self._k, self._k))
                summation = np.zeros((self._k, 1))

                #for every non-nan element in row we take corresponding column of Y and make manipulations
                for arg in args:
                    summation_inv = summation_inv + (X[:, arg].reshape(-1,1)).dot(X[:, arg].reshape(1, -1))
                    summation = summation + column[arg] * X[:, arg].reshape(-1,1)

                #update the corresponding column of Y
                new_Y = solve(summation_inv + self._lambda * np.eye(self._k), summation)
                column_diff_norm += np.linalg.norm(Y[:,j] - new_Y) / np.linalg.norm(Y[:,j])
                Y[:,j] = new_Y.reshape(-1)

            self._errors.append(column_diff_norm)
            if column_diff_norm < 1e-5:
                break

        #save the results as the attribute of class
        self._X = X
        self._Y = Y
    
    def _update_proj(self):
        """
        Set self._proj to be a sparse matrix containing a projection of a current approximation
        on the set of known values
        """
        proj_data = np.empty(self._nonzero[0].size)

        for i in range(self._nonzero[0].size):
            proj_data[i] = self._X.T[self._nonzero[0][i], :].dot(self._Y[:, self._nonzero[1][i]])

        self._proj = sparse.csr_matrix((proj_data, self._nonzero), self._data.shape)
    
    def get_factors(self):
        """
        Returns X and Y - factorization of matrix, used in fit method
        """
        return self._X, self._Y
        
    def predict(self):
        """
        Returns the product of factorization matrices X and Y: X^T*Y
        """
        return (self._X.T).dot(self._Y)
    
    def get_errors(self):
        """
        Attribute _errors contains sum of squares of differences between prior and updated columns 
        of X and Y on each iteration.
        Method get_errors returns these errors
        """
        return self._errors
    
    def get_bias(self):
        """
        Attribute _bias contains sum of squares of differences between observed elements in fitted matrix
        and corresponding elements of X^T*Y
        Method get_bias returns these errors
        """
        return self._bias
