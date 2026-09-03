# AI Attack Detection





## Project Overview



This project implements a machine learning-based system for network attack detection and classification.









**The system uses a two-layer classification approach:**



* Layer 1: Binary classification to identify whether network traffic is Normal or an Attack.
* Layer 2: Multi-class classification to identify the type of attack.

&#x20;     Machine Learning Pipeline











**The project uses the following steps:**



1- Load and merge the datasets.

2- Preprocess the data.

3- Standardize features using StandardScaler.

4- Remove low-variance features using VarianceThreshold.

5- Select important features using SelectFromModel with a Random Forest classifier.

6- Handle class imbalance using SMOTE.



7- Train an ensemble model using:



* XGBoost
* LightGBM
* CatBoost



8- Combine the models using VotingClassifier.

9- Evaluate the trained models using several evaluation methods.











**The models are evaluated using:**



* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Stratified 5-Fold Cross-Validation
* Inference Time



The generated results and visualizations are stored in the results/ directory.









**Technologies Used:**



* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* LightGBM
* CatBoost
* Matplotlib
* Seaborn
* Google Colab









**How to Run:**



The complete experiment can be run using the notebook located in the notebooks/ directory.








**Dataset:**



This project uses network traffic datasets for cyber attack detection and classification.

The datasets contain both normal network traffic and different types of cyber attacks and are used for training and evaluating the proposed two-layer machine learning system:

* Layer 1: Binary classification (Normal vs. Attack)
* Layer 2: Multi-class classification of different attack types


  
**Dataset Availability:**



The datasets are not included in this repository because some of the original CSV files exceed GitHub's file size limit.
The dataset was used locally during the development, training, and evaluation of the machine learning models.





**The required Python libraries are listed in:**



requirements.txt

