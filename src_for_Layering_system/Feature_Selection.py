
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def create_feature_selector():
    """
    Create a Random Forest-based feature selector.

    Features with importance greater than the mean
    importance are retained.
    """

    rf_selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        threshold="mean"
    ).set_output(transform="pandas")

    return rf_selector




