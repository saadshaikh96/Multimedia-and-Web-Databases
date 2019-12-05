import numpy as np
from sklearn.decomposition import TruncatedSVD

def get_svd_decomposition(data, k):
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
    svd.fit(data)
    X_svd = svd.transform(data)
    vt = svd.components_
    return X_svd, vt

def get_svd_transform(vt, data):
    return np.dot(data, vt.T)