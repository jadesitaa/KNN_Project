
import pandas as pd
df = pd.read_csv('/content/Titanic-Dataset.csv')

# เช็คแล้วไม่มีค่าว่าง
print(df.isnull().sum())
#แปลง column ที่ไม่ใช่ตัวเลข
print(df.dtypes)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
