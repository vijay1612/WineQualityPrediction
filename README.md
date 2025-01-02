# Wine Quality Prediction

This project aims to predict the quality of red wine based on various physicochemical attributes. The dataset contains several features related to wine chemistry, and the goal is to classify the wine as either “good” or “bad” quality. The model uses machine learning techniques, such as Logistic Regression, Support Vector Machines (SVM), and Neural Networks, to predict the quality of the wine.

Project Overview

Wine quality prediction is a classic supervised classification problem where the goal is to predict wine quality based on its chemical properties. The dataset contains features such as acidity, alcohol content, pH, sulfur dioxide levels, and other characteristics that influence wine quality.

# Key Features
	•	Predict Wine Quality: Classify wines into two categories: good and bad.
	•	Data Preprocessing: Handle missing values, categorical data, and normalize the dataset.
	•	Machine Learning Models: Implement models including Logistic Regression, Stochastic Gradient Descent (SGD), Support Vector Machines (SVM), Decision Trees, Random Forest, and Multi-layer Perceptron (MLP).
	•	Evaluation Metrics: Use classification accuracy, precision, recall, F1-score, and confusion matrix to evaluate the performance of models.

# Dataset

The dataset used is Wine Quality Dataset, which consists of 1599 red wine samples with 11 input features:
	•	fixed acidity
	•	volatile acidity
	•	citric acid
	•	residual sugar
	•	chlorides
	•	free sulfur dioxide
	•	total sulfur dioxide
	•	density
	•	pH
	•	sulphates
	•	alcohol

The target variable is quality, which is an integer value ranging from 3 to 8. This variable is transformed into binary categories: bad (0) and good (1).

Data Summary
	•	The dataset has 1599 entries, and the quality distribution is split between good and bad categories.
	•	Missing data: There are no missing values in the dataset.

Data Visualization
	•	Scatter plots: To visualize the relationship between features and wine quality.
	•	Correlation Heatmap: To see the correlation between various features and their relationship with wine quality.
	•	Bar plots: For visualizing how different features vary with respect to wine quality.

Data Preprocessing
	•	Feature Engineering: Some features such as volatile acidity, chlorides, and density were found to have low correlation with wine quality and were dropped for improved model performance.
	•	Label Encoding: The quality column was converted into binary labels (0 for bad quality, 1 for good quality).
	•	Feature Scaling: Standardization was performed using StandardScaler to scale features to a similar range.

Technical Details

Libraries Used
	•	Pandas: Data manipulation and analysis.
	•	NumPy: Numerical operations and data handling.
	•	Matplotlib: Data visualization and plotting.
	•	Seaborn: Advanced visualization for bar plots and heatmaps.
	•	Scikit-learn: Machine learning models, preprocessing, and evaluation.
	•	Keras & TensorFlow: For building and training neural network models.

Machine Learning Models
	1.	Logistic Regression:
	•	A simple linear model that was trained to predict whether the wine quality is good or bad.
	•	Training Accuracy: 74.81%
	•	Testing Accuracy: 72.25%
	2.	Stochastic Gradient Descent (SGD):
	•	An iterative optimization technique for training machine learning models.
	•	Testing Accuracy: 65%
	3.	Support Vector Machine (SVM):
	•	A model that maximizes the margin between classes.
	•	Tuned hyperparameters using GridSearchCV to find the best combination of C, gamma, and kernel.
	•	Testing Accuracy: 73%
	4.	Decision Tree:
	•	A tree-like model for classification. High training accuracy but potential overfitting.
	•	Testing Accuracy: 73.5%
	5.	Random Forest:
	•	An ensemble of decision trees that improves performance by aggregating the predictions.
	•	Testing Accuracy: 79.5%
	6.	Multilayer Perceptron (MLP):
	•	A type of neural network with one or more hidden layers.
	•	Testing Accuracy: 76%
	7.	Artificial Neural Network (ANN):
	•	A deep learning model built with Keras and TensorFlow, trained with multiple epochs.
	•	Training Accuracy: 93.74%
	•	Testing Accuracy: 76%

Hyperparameter Tuning
	•	SVM: Tuned using GridSearchCV to find the optimal combination of C, gamma, and kernel.
	•	MLP: Optimized the number of epochs and batch size to achieve better performance.

Model Performance Evaluation
	•	Confusion Matrix: Analyzed the model’s classification performance (precision, recall, F1-score).
	•	Cross-Validation: Used 10-fold cross-validation to evaluate the models’ performance on unseen data.

Future Work
	•	Explore deep learning models further by increasing the complexity and adding more hidden layers.
	•	Investigate ensemble methods like Gradient Boosting or XGBoost for potentially better performance.
	•	Implement feature selection techniques to reduce dimensionality and improve model efficiency.

Conclusion

This project demonstrates how different machine learning algorithms can be applied to predict wine quality based on various features.
It highlights the importance of feature selection, model tuning, and evaluation metrics in building accurate predictive models.
To Use This Repository:
	1.	Clone the repository:
 git clone https://github.com/vijay1612/wine-quality-prediction.git
 Install the dependencies:
 pip install -r requirements.txt
 	3.	Run the project:
	•	Download the wine quality dataset (e.g., winequality-red.csv).
	•	Place the dataset in the appropriate directory.
	•	Run the script to train the model and evaluate the results.
