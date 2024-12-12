import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='Class')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def reduce_and_classify(name, reducer, X_train, X_test, y_train, y_test):
    # Reduksi Dimensi
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)

    # Klasifikasi dengan Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_test_reduced)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=wine.target_names)

    return X_train_reduced, X_test_reduced, cm_display

    pca = PCA(n_components=2)
X_train_pca, X_test_pca, cm_pca = reduce_and_classify("PCA", pca, X_train, X_test, y_train, y_test)


svd = TruncatedSVD(n_components=2)
X_train_svd, X_test_svd, cm_svd = reduce_and_classify("SVD", svd, X_train, X_test, y_train, y_test)


isomap = Isomap(n_components=2)
X_train_isomap, X_test_isomap, cm_isomap = reduce_and_classify("ISOMAP", isomap, X_train, X_test, y_train, y_test)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette="Set1")
plt.title("Visualisasi Data Asli")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title='Class', loc='upper left')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette="Set1")
plt.title("PCA Dimensionality Reduction")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Class', loc='upper left')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_svd[:, 0], y=X_train_svd[:, 1], hue=y_train, palette="Set1")
plt.title("SVD Dimensionality Reduction")
plt.xlabel("SVD1")
plt.ylabel("SVD2")
plt.legend(title='Class', loc='upper left')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_isomap[:, 0], y=X_train_isomap[:, 1], hue=y_train, palette="Set1")
plt.title("ISOMAP Dimensionality Reduction")
plt.xlabel("ISOMAP1")
plt.ylabel("ISOMAP2")
plt.legend(title='Class', loc='upper left')
plt.show()

print("\nFitur Awal:")
print(X.head())

print("\nFitur Setelah PCA (2 Komponen):")
X_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
X_pca_df['Class'] = y_train.reset_index(drop=True)  # Pastikan label sesuai dengan data
print(X_pca_df.head())

print("\nFitur Setelah SVD (2 Komponen):")
X_svd_df = pd.DataFrame(X_train_svd, columns=['SVD1', 'SVD2'])
X_svd_df['Class'] = y_train.reset_index(drop=True)  # Pastikan label sesuai dengan data
print(X_svd_df.head())


print("\nFitur Setelah ISOMAP (2 Komponen):")
X_isomap_df = pd.DataFrame(X_train_isomap, columns=['ISOMAP1', 'ISOMAP2'])
X_isomap_df['Class'] = y_train.reset_index(drop=True)  # Pastikan label sesuai dengan data
print(X_isomap_df.head())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
cm_pca.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix - PCA")

# SVD
cm_svd.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix - SVD")

# ISOMAP
cm_isomap.plot(cmap=plt.cm.Blues, ax=axes[2], colorbar=False)
axes[2].set_title("Confusion Matrix - ISOMAP")

plt.tight_layout()
plt.show()