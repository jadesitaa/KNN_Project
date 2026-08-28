
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('/content/Titanic-Dataset.csv')

# เช็คแล้วมีค่าว่าง
print(df.isnull().sum())

# Handle missing values: Fill Age with median, Embarked with mode. Drop Cabin (too sparse).
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
#ตัดคอลัมน์ที่เป็นข้อความและไม่จำเป็นทิ้งจาก X 
X = X.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'], errors='ignore')

#แบ่งชุดข้อมูล (Train/Test Split) ใหม่อีกครั้ง
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#ปรับสเกลข้อมูล (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build KNN Model (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
