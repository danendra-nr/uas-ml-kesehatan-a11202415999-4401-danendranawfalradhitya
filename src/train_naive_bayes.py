from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    """Train Gaussian Naive Bayes classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model
