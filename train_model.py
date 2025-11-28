from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Irisデータセットの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# KNNモデルの作成 (近傍点=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# モデルをファイルに保存
joblib.dump(model, "model.pkl")
print("model.pkl を保存しました。")
