
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/Titanic-Dataset.csv')

# เช็คแล้วมีค่าว่าง
print(df.isnull().sum())
# Handle missing values: Fill Age with median, Embarked with mode. Drop Cabin (too sparse).
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Split data
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
