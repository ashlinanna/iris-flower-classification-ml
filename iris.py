

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             classification_report, confusion_matrix)

# ─────────────────────────────────────────────
# 1. Load & Explore
# ─────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 60)
print("TASK 1 – IRIS FLOWER CLASSIFICATION")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Classes       : {list(iris.target_names)}")
print(f"\nClass distribution:\n{df['species'].value_counts().to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nDescriptive stats:\n{df.describe().round(2).to_string()}")

# ─────────────────────────────────────────────
# 2. Visualisations  (4-panel figure)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Dataset – Exploratory Data Analysis", fontsize=15, fontweight='bold')

palette = {'setosa': '#4CAF50', 'versicolor': '#2196F3', 'virginica': '#F44336'}

# (a) Sepal scatter
ax = axes[0, 0]
for sp, grp in df.groupby('species'):
    ax.scatter(grp['sepal length (cm)'], grp['sepal width (cm)'],
               label=sp, alpha=0.8, edgecolors='k', linewidth=0.4,
               color=palette[sp], s=70)
ax.set_xlabel('Sepal Length (cm)'); ax.set_ylabel('Sepal Width (cm)')
ax.set_title('Sepal: Length vs Width'); ax.legend(); ax.grid(True, alpha=0.3)

# (b) Petal scatter
ax = axes[0, 1]
for sp, grp in df.groupby('species'):
    ax.scatter(grp['petal length (cm)'], grp['petal width (cm)'],
               label=sp, alpha=0.8, edgecolors='k', linewidth=0.4,
               color=palette[sp], s=70)
ax.set_xlabel('Petal Length (cm)'); ax.set_ylabel('Petal Width (cm)')
ax.set_title('Petal: Length vs Width'); ax.legend(); ax.grid(True, alpha=0.3)

# (c) Histograms – petal length
ax = axes[1, 0]
for sp, grp in df.groupby('species'):
    ax.hist(grp['petal length (cm)'], bins=15, alpha=0.65,
            label=sp, color=palette[sp], edgecolor='k', linewidth=0.4)
ax.set_xlabel('Petal Length (cm)'); ax.set_ylabel('Count')
ax.set_title('Histogram – Petal Length'); ax.legend(); ax.grid(True, alpha=0.3)

# (d) Box plot – all features
ax = axes[1, 1]
df_melt = df.melt(id_vars='species', var_name='feature', value_name='value')
species_list = df['species'].cat.categories.tolist()
color_list   = [palette[s] for s in species_list]
sns.boxplot(data=df_melt, x='feature', y='value', hue='species',
            palette=palette, ax=ax, linewidth=0.8)
ax.set_title('Box Plot – All Features')
ax.set_xlabel('Feature'); ax.set_ylabel('Value (cm)')
ax.tick_params(axis='x', rotation=15)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] iris_eda.png")

# ─────────────────────────────────────────────
# 3. Train / Test Split & Scaling
# ─────────────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTrain size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. Train Three Classifiers
# ─────────────────────────────────────────────
models = {
    'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression'       : LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree'             : DecisionTreeClassifier(max_depth=4, random_state=42),
}

results = {}
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    results[name] = {'model': model, 'y_pred': y_pred, 'accuracy': acc, 'precision': prec}
    print(f"\n{'─'*40}\n{name}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─────────────────────────────────────────────
# 5. Confusion Matrix plot for best model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_pred = results[best_name]['y_pred']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices – All Three Classifiers", fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=iris.target_names, yticklabels=iris.target_names,
                linewidths=0.5, cbar=False)
    ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.1f}%", fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('iris_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] iris_confusion.png")
print(f"\n✅ Best model : {best_name}  (Accuracy: {results[best_name]['accuracy']*100:.1f}%)")