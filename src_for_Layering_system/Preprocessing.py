

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold



scaler = StandardScaler().set_output(transform="pandas")


var_thresh = VarianceThreshold(
    threshold=0.01
).set_output(transform="pandas")