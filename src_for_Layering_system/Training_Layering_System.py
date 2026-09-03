

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import  VotingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier



# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Smote process to increase number of training sampels.
smote_layer1 = SMOTE(sampling_strategy={0: 52102}, random_state=42)

# Smote process to increase number of training sampels.
smote_layer2 = SMOTE(sampling_strategy= { 3:3270 , 4:3270 , 2:1500 }, random_state=42)


# Build an ensemble model using hard voting from three different classifiers
# Each model votes, and the final prediction is the majority class
voting_model = VotingClassifier([
    ('xgb', XGBClassifier(n_estimators=100,random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=100, random_state=42)),
    ('cat', CatBoostClassifier(iterations=100, random_state=42, verbose=0) )
], voting='soft')   # soft = average probabilities
  

# Full pipeline with full steps.
model = Pipeline([
    ('scaler', scaler),
    ('variance_filter', var_thresh),
    ('feature_selector', rf_selector),
    ('smote', smote),
    ('ensemble', voting_model)
])


# learn model.
model.fit(X_train, y_train)

