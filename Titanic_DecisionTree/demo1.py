import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv("test.csv")

print(train_data.isnull().sum())

train_data['Age'].fillna(train_data['Age'].median(),inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0],inplace=True)
train_data.drop('Cabin',axis=1,inplace=True)

print(test_data.isnull().sum())
test_data['Age'].fillna(test_data['Age'].median(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(),inplace=True)

train_data['Sex'] = train_data['Sex'].map({'male':0,'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'C':0,'Q':1,'S':2})

test_data['Sex'] = test_data['Sex'].map({'male':0,'female':1})
test_data['Embarked'] = test_data['Embarked'].map({'C':0,'Q':1,'S':2})

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = train_data[features]
y = train_data['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train,y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f'准确率: {accuracy:.4f}')

print(classification_report(y_test,y_pred))

conf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=['Dead','Survived'],yticklabels=['Dead','Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

x_test_data = test_data[features]
test_predictions = dt_classifier.predict(x_test_data)

submission = pd.DataFrame({
    'PassengerId':test_data['PassengerId'],
    'Survived':test_predictions
})

submission.to_csv('submission.csv',index=False)