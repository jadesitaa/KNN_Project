import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {
    'Color': ['Red', 'Red', 'Red', 'Yellow', 'Yellow', 'Yellow', 'Yellow', 'Yellow', 'Red', 'Red'],
    'Type': ['Sport', 'Sport', 'Sport', 'Sport', 'Sport', 'SUV', 'SUV', 'SUV', 'SUV', 'Sport'],
    'Origin': ['Domestic', 'Domestic', 'Domestic', 'Domestic', 'Imported', 'Imported', 'Imported', 'Domestic', 'Imported', 'Imported'],
    'Stolen': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
}
df = pd.DataFrame(data)
X = pd.get_dummies(df[['Color', 'Type', 'Origin']])
y = df['Stolen']

# สร้างและเทรนโมเดล
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)
print("Model trained successfully")

sample = pd.DataFrame({
    'Color_Red': [1], 'Color_Yellow': [0],
    'Type_SUV': [1], 'Type_Sport': [0],
    'Origin_Domestic': [1], 'Origin_Imported': [0]
})

# จัดเรียงคอลัมน์ให้ตรงกับตอนเทรน
sample = sample.reindex(columns=X.columns, fill_value=0)

prediction = clf.predict(sample)
print(f"ผลการทำนายสำหรับรถ (Red, SUV, Domestic): {prediction[0]}")

plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], filled=True)
plt.show()
