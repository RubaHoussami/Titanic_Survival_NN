import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def BuildDataset():
    # Define the features to use
    features = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Load the titanic dataset
    data = pd.read_csv('data/Titanic-Dataset.csv', usecols=features)

    data = data.dropna(subset=['Survived'])  # Drop rows without survival info (unlabeled)

    # Fill missing values
    age_mean = data['Age'].mean()
    data['Age'] = data['Age'].fillna(age_mean)
    embarked_mode = data['Embarked'].mode()[0]
    data['Embarked'] = data['Embarked'].fillna(embarked_mode)
    fare_mean = data['Fare'].mean()
    data['Fare'] = data['Fare'].fillna(fare_mean)

    # Encode categorical variables using one-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    # Separate features and target
    X = data.drop(columns=['Survived']).values
    y = data['Survived'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test
