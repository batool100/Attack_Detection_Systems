
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import time




def Average_Inference_Time(X_test , Model , Type):
  ''' This to print average inference
       time in prediction for models.

    Parameters:
    - X_test: Test data features.
    - Model: Trained model.
    - Type: name of model
  '''

  # Take a random sample of 100 instances from the test set for inference time evaluation
   samples = X_test.sample(min(100, len(X_test)), random_state=42)

  # Initialize list to store time taken for each prediction
  times = []

  # Loop through each sample and measure the time taken to predict
  for i in range(len(samples)):
    start = time.time()
    Model.predict(samples.iloc[[i]])
    end = time.time()
    times.append(end - start)

  # Calculate and display the average prediction (inference) time
  avg_time = sum(times) / len(times)
  print(f"\n\nAverage inference time per sample {Type} (100 samples): {avg_time} seconds.\n\n")






def plot_pie_chart(X_train, y_train,smote, name):
    """
    plot pie chart for classes distribution.

    Parameters:
    - X_train: train data features.
    - y_train: True labels.
    - smote: smote type used.
    - name: name of system used. 
    """

    # Perform SMOTE on the training set
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Count the number of samples per class before and after SMOTE
    before_counts = Counter(y_train)
    after_counts = Counter(y_train_smote)

    # Plot class distributions using Pie Charts
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Pie chart before SMOTE
    ax[0].pie(before_counts.values(), labels=[f'{k} ({v})' for k,v in before_counts.items()],
           autopct='%1.1f%%', startangle=90, colors=['#a587ca', '#fe797b','#8fe968', '#ffea56','#36cedc'])
    ax[0].set_title(f"Before SMOTE( {name} )")

    # Pie chart after SMOTE
    ax[1].pie(after_counts.values(), labels=[f'{k} ({v})' for k,v in after_counts.items()],
           autopct='%1.1f%%', startangle=90, colors=['#a587ca', '#fe797b','#8fe968', '#ffea56','#36cedc'])
    ax[1].set_title(f"After SMOTE( {name} )")

    # Display the plots
    plt.tight_layout()
    plt.show()






def plot_confusion_matrix_voting(model, model_name, X_test, y_test, class_names, lab, title="Confusion Matrix"):
    """
    Plot a single confusion matrix for a pipeline-based voting model.

    Parameters:
    - model: Trained pipeline model with a voting classifier.
    - model_name: Name of the model for labeling the plot.
    - X_test: Test data features.
    - y_test: True labels.
    - class_names: List of class labels for the confusion matrix display.
    - lab: List of labels for the confusion matrix display.
    - title: Title for the confusion matrix plot.
    """

    # Predict using the pipeline model
    y_pred = model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels= lab)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Greens', colorbar=False)
    ax.set_title(f"{title} - {model_name}")
    plt.tight_layout()
    print("\n\n\n")
    plt.show()
    print("\n\n\n")




def Evaluation_Printing(X_data , y_data , y_test , pred , my_pipeline , st):
  ''' This to print evaluation of work
        of models by several measures.

    Parameters:
    - X_data: Train data features.
    - y_data: True labels.
    - y_test: True labels.
    - pred: prediction by model
    - my_pipeline: pipeline model.
    - st: name of system used.
  '''

  # Print the model's accuracy on the test set
  print(f"\n\n\nAccuracy of {st} :", accuracy_score(y_test, pred)*100 ,"%",'\n')

  # Show detailed classification metrics: precision, recall, f1-score for each class.
  print(f"\nClassification Report for {st}:")
  print(classification_report(y_test, pred))

  # Evaluate the model using 5-Fold Stratified Cross-Validation.
  CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  cv_scores = cross_val_score(my_pipeline, X_data, y_data, cv=CV, scoring='accuracy')
  print(f"\n\nAccuracy by Cross Validation of {st} :\n", cv_scores)


  # Visualize accuracy across folds.
  # Similar scores across folds indicate stable cross-validation performance.
  print("\n\nA drawing showing the absence of overfitting through cross validation:\n")
  mean_score = np.mean(cv_scores)
  std_score = np.std(cv_scores)

  plt.figure(figsize=(8, 5))
  plt.plot(range(1, 6), cv_scores, marker='o', label='Validation Accuracy')
  plt.axhline(mean_score, color='green', linestyle='--', label=f'Mean Accuracy = {mean_score:.2f}')
  plt.fill_between(range(1, 6), mean_score - std_score, mean_score + std_score, color='green', alpha=0.2)

  plt.title("Cross-Validation Accuracy Across Folds")
  plt.xlabel("Fold Number")
  plt.ylabel("Accuracy")
  plt.ylim(0.7, 1.0)
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()





def plot_confusion_matrix_single(model, X_test, y_test, class_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix for a single classification model.

    Parameters:
    - model: Trained classification model.
    - X_test: Test data features.
    - y_test: True labels.
    - class_names: List of class labels (optional).
    - title: Title of the plot.
    """
 
    # Predict using the pipeline model
    y_pred = model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay( confusion_matrix=cm , display_labels= class_names )
    disp.plot(cmap='Greens', values_format='d')
    plt.title(title)
    plt.tight_layout()
    print("\n\n\n")
    plt.show()
    print("\n\n\n")



