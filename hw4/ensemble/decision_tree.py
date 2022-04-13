import numpy as np
import random


class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        sample_weights=sample_weights/np.sum(sample_weights)
        #print(y.shape,sample_weights.shape)
        p1=np.matmul(y,sample_weights)
        if abs(p1-1)<1e-5 or abs(p1)<1e-5:
            entropy=0
        else:
            entropy=-p1*np.log2(p1)-(1-p1)*np.log2(1-p1)
        # end answer
        #print(entropy,p1,y)
        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0
        # YOUR CODE HERE
        # begin answer
        fvalue=[]
        info_gain=self.entropy(y,sample_weights)
        #print(info_gain)
        fvalue=np.unique(X[:,index])
        
        #print(fvalue)
        for value in fvalue:
            dex=np.where(X[:,index]==value)[0]
            wv=np.sum(sample_weights[dex])
            w=np.sum(sample_weights)
            info_gain=info_gain-wv/w*self.entropy(y[dex],sample_weights[dex])
        # end answer
        #print(dex.shape,y.shape)
        return info_gain

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        info_gain_ratio = 0
        split_information = 0.0
        # YOUR CODE HERE
        # begin answer
        fvalue=np.unique(X[:,index])
        for value in fvalue:
            dex=np.where(X[:,index]==value)[0]
            rat=np.sum(sample_weights[dex])/np.sum(sample_weights)
            if rat>0:
                split_information=split_information-rat*np.log2(rat)
        info_gain_ratio=self._information_gain(X,y,index,sample_weights)/split_information
        # end answer
        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        p1=np.matmul(sample_weights,y)
        gini=2*p1*(1-p1)
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1
        # YOUR CODE HERE
        # begin answer
        fvalue=np.unique(X[:,index])
        for value in fvalue:
            dex=np.where(X[:,index]==value)[0]
            rat=np.sum(sample_weights[dex])/np.sum(sample_weights)
            new_impurity=new_impurity-rat*rat
        # end answer
        return new_impurity

    def _split_dataset(self, X, y, index, value, sample_weights):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        dex=np.where(X[:,index]==value)[0]
        X=np.delete(X,index,axis=1)
        sub_X=X[dex,:]
        sub_y=y[dex]
        sub_sample_weights=sample_weights[dex]
        # end answer
        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        max=-1
        if self.sample_feature==False:
            for i in range(X.shape[1]):
                a=self.criterion(X,y,i,sample_weights)
                #print(a)
                if a>max:
                    best_feature_idx=i
                
                    max=a
            #print(best_feature_idx)
            # end answer
        else:
            for i in np.random.choice(X.shape[1],X.shape[1]//2,replace=False):
                a=self.criterion(X,y,i,sample_weights)
                #print(a)
                if a>max:
                    best_feature_idx=i
                
                    max=a
        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        rat=np.matmul(sample_weights,y)
        if rat>0.5:
            majority_label=1
        else:
            majority_lable=0
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf, we use the majority_label by calling majority_vote.
                mytree = {
                    'titile': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X. Then use `_split_dataset` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful. build_tree is recursive.
        # begin answer
        N,D=X.shape
        if D==1:
            a=dict()
            value=np.unique(X)
            for i in value:
                idx=np.where(X[:,0]==i)
                a[i]=self.majority_vote(y[idx],sample_weights[idx])
            mytree[feature_names[0]]=a
        elif N<self.min_samples_leaf:
            mytree=self.majority_vote(y,sample_weights)
        elif depth>self.max_depth:
            mytree=self.majority_vote(y,sample_weights)
        else:
            best_idx=self._choose_best_feature(X,y,sample_weights)
            #print(best_idx,X.shape,y.shape)
            a=dict()
            value=np.unique(X[:,best_idx])
            
            for i in value:
                sub_X, sub_y, sub_sample_weights=self._split_dataset( X, y, best_idx, i, sample_weights)
                a[i]=self._build_tree(sub_X, sub_y, np.delete(feature_names,best_idx), depth=depth+1, sample_weights=sub_sample_weights)
            mytree[feature_names[best_idx]]=a
                
        #print(depth)
        # end answer
        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x,i):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            if type(tree)!=dict:
                return tree
            else:
                feature=list(tree.keys())
                #print(tree)
                #print(x)
                #print(feature)
                
                if x.iloc[0][feature[0]] not in tree[feature[0]].keys():
                    return int(random.uniform(0,2))
                else:
                    return _classify(tree[feature[0]][x.iloc[0][feature[0]]],x.drop([feature[0]],axis=1),i)
            # end answer

        # YOUR CODE HERE
        # begin answer
        N=np.array(X).shape[0]
        pre=np.zeros(N)
        
        #print(self._tree)
        
        for i in range(N):
            pre[i]=_classify(self._tree,X.iloc[[i]],i)
        return pre
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
