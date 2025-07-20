#step 1 data preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load dataset
df = pd.read_csv("Titanic-Dataset.csv")

#drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#fill the missing values
df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)

#encode categorical variables into numeric values 
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

#separate features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

#show
print(X.head())
print(y.head())


#step 2 logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#evaluate model performance
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#step 3 decision tree model
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
#evaluate decision tree performance
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))


#step 4 visualization and analysis

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

plt.ioff() 


#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show(block=True)  
plt.close()


#Print Predictions
print("\nSample Predictions:")
print(pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10]}))
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
plt.figure()
plt.barh(X.columns, rf_model.feature_importances_, color='skyblue')
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.show(block=True)
plt.close()






