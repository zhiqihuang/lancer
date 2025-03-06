import numpy as np
from sklearn.linear_model import SGDClassifier
import sklearn

EVAL_CLF_PARAMS = {"loss": "log_loss", "tol": 1e-4, "iters_no_change": 15, "alpha": 1e-4, "max_iter": 25000}
NUM_CLFS_IN_EVAL = 1 # change to 1 for large dataset / high dimensionality

def init_classifier():

    return SGDClassifier(loss=EVAL_CLF_PARAMS["loss"], fit_intercept=True, max_iter=EVAL_CLF_PARAMS["max_iter"], tol=EVAL_CLF_PARAMS["tol"], n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
                        n_jobs=32, alpha=EVAL_CLF_PARAMS["alpha"])
                        

def get_score(X_train, y_train, X_dev, y_dev):
    loss_vals = []
    train_accs = []
    dev_accs = []
    
    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_dev)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        train_accs.append(clf.score(X_train, y_train))
        dev_accs.append(clf.score(X_dev, y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], train_accs[i], dev_accs[i]