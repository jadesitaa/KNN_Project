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
