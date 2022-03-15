from os import spawnl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, 
    precision_score, recall_score, confusion_matrix,
    roc_curve, roc_auc_score
) 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


data_url = 'https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/diabetes.csv'
diabetes = pd.read_csv(data_url)
diabetes.head()

# separate features from label
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
label = 'Diabetic'
X, y = diabetes[features].values, diabetes[label].values

# Print features for first 4 patients
for n in range(0,4):
    print('Patient', str(n+1), '\n Features:', list(X[n]), '\n Label:', y[n])

# Plot features by label
for col in features:
    diabetes.boxplot(column=col, by='Diabetic', figsize=(6,6))
    plt.title(col)
plt.show()

# split train test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape

# model evaluation function
def eval_model(model):
    
    print(model)
    predictions = model.predict(X_test)
    print('Predicted labels: ', predictions)
    print('Actual Labels:    ', y_test)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print('Overall Precision: ', precision_score(y_test, predictions))
    print('Overall Recall: ', recall_score(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    print(cm)

    # plot the ROC curve
    y_scores = model.predict_proba(X_test)
    print(y_scores)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

    fig = plt.figure(figsize=(6,6))
    # add diagonal line
    plt.plot([0,1], [0,1], 'k--')

    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.show()

    auc = roc_auc_score(y_test, y_scores[:,1])
    print('AUC: ' + str(auc))


# train logistic regression model
reg = 0.01
model = LogisticRegression(C=1/reg, solver='liblinear').fit(X_train, y_train)
eval_model(model)

# retrain with some preprocessing of the data
numeric_features = [0,1,2,3,4,5,6]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_features = [7]
categorical_transformer = Pipeline(steps=[
    ('onhot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    ('logregression', LogisticRegression(C=1/reg, solver='liblinear'))])

model = pipeline.fit(X_train, y_train)
eval_model(model)

# train a random forest model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('randforest', RandomForestClassifier(n_estimators=100))])

model = pipeline.fit(X_train, y_train)
eval_model(model)

#### Multiclass classification ####
data_url = 'https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/penguins.csv'
penguins = pd.read_csv(data_url)
sample = penguins.sample(10)
sample

penguin_classes = ['Adelie', 'Gentoo', 'Chinstrap']
print(sample.columns[0:5].values, 'SpeciesName')
for index, row in penguins.sample(10).iterrows():
    print('[', row[0], row[1], row[2], row[3], int(row[4]), ']', penguin_classes[int(row[4])])

penguins.isnull().sum()
penguins[penguins.isnull().any(axis=1)]
penguins = penguins.dropna()

# Plot features relationships with label
penguin_features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
penguin_label = 'Species'
for col in penguin_features:
    penguins.boxplot(column=col, by=penguin_label, figsize=(6,6))
    plt.title(col)
plt.show()

penguins_X, penguins_y = penguins[penguin_features].values, penguins[penguin_label].values
x_penguin_train, x_penguin_test, y_penguin_train, y_penguin_test = train_test_split(penguins_X, penguins_y,
    test_size=0.3,
    random_state=0,
    stratify=penguins_y)
x_penguin_train.shape

# train logistic regression
reg = 0.1

multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(x_penguin_train, y_penguin_train)
print(multi_model)
penguin_predictions = multi_model.predict(x_penguin_test)
print('Predicted labels: ', penguin_predictions[:15])
print('Actual labels   : ' ,y_penguin_test[:15])
print(classification_report(y_penguin_test, penguin_predictions))
print("Overall Accuracy:",accuracy_score(y_penguin_test, penguin_predictions))
print("Overall Precision:",precision_score(y_penguin_test, penguin_predictions, average='macro'))
print("Overall Recall:",recall_score(y_penguin_test, penguin_predictions, average='macro'))
mcm = confusion_matrix(y_penguin_test, penguin_predictions)
print(mcm)

plt.imshow(mcm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=45)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.show()

penguin_prob = multi_model.predict_proba(x_penguin_test)

fpr = {}
tpr = {}
thresh = {}
for i in range(len(penguin_classes)):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_penguin_test, penguin_prob[:,i], pos_label=i)

plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=penguin_classes[0] + ' vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=penguin_classes[1] + ' vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=penguin_classes[2] + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

auc = roc_auc_score(y_penguin_test,penguin_prob, multi_class='ovr')
print('Average AUC:', auc)

# use a preprocessing pipeline
feature_columns = [0,1,2,3]
feature_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('preprocess', feature_transformer, feature_columns)])

# Create training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', SVC(probability=True))])


# fit the pipeline to train a linear regression model on the training set
multi_model = pipeline.fit(x_penguin_train, y_penguin_train)

# Get predictions from test data
penguin_predictions = multi_model.predict(x_penguin_test)
penguin_prob = multi_model.predict_proba(x_penguin_test)

# Overall metrics
print("Overall Accuracy:",accuracy_score(y_penguin_test, penguin_predictions))
print("Overall Precision:",precision_score(y_penguin_test, penguin_predictions, average='macro'))
print("Overall Recall:",recall_score(y_penguin_test, penguin_predictions, average='macro'))
print('Average AUC:', roc_auc_score(y_penguin_test,penguin_prob, multi_class='ovr'))

# Confusion matrix
plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=45)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()


# Make predictions
x_new = np.array([[50.4, 15.3, 224, 5550]])

penguin_pred = multi_model.predict(x_new)[0]
penguin_classes[penguin_pred]
