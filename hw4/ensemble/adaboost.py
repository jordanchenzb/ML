import copy
import numpy as np


class Adaboost:
    '''Adaboost Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]
        self._alphas = [1 for _ in range(n_estimator)]

    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        N=X.shape[0]
        sample_weights=np.array([1/N for i in range(N)])
        
        for i in range(self.n_estimator):
            self._estimators[i].fit(X,y,sample_weights)
            y_p=self._estimators[i].predict(X)
            err=np.matmul(sample_weights,y!=y_p)/np.sum(sample_weights)
            self._alphas[i]=np.log((1-err)/err)
            sample_weights=sample_weights*np.exp(self._alphas[i]*(y_p!=y))
            
        # end answer
        return self

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)
        # YOUR CODE HERE
        # begin answer
        labels=np.array([self._estimators[j].predict(X) for j in range(self.n_estimator)])
        labels[labels==0]=-1
        alphas=np.array(self._alphas)
        y_pred=np.sign(np.matmul(alphas,labels))
        y_pred[y_pred==-1]=0;
        # end answer
        return y_pred
