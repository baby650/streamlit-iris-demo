import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 日本語フォント設定（Streamlit Cloud で動作するフォント）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
# 日本語フォント設定 (デスクトップ上)
# plt.rcParams['font.family'] = 'MS Gothic'
# モデル読み込み
model = joblib.load("model.pkl")

st.set_page_config(page_title="Iris Classification", layout="centered")

st.title("アヤメの品種分類 (KNN)")

# 入力フォーム
sepal_length = st.number_input("がく片の長さ", 0.0, 10.0, 5.1)
sepal_width = st.number_input("がく片の幅", 0.0, 10.0, 3.5)
petal_length = st.number_input("花弁の長さ", 0.0, 10.0, 1.4)
petal_width = st.number_input("花弁の幅", 0.0, 10.0, 0.2)

species = ["setosa", "versicolor", "virginica"]

if st.button("予測"):
    pred_id = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    st.success(f"予測結果: **{species[pred_id]}** (class {pred_id})")

    # --- 視覚化 ---
    st.subheader("散布図")

    iris = load_iris()
    X = iris.data
    y = iris.target

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["blue", "orange", "green"]

    for i, color in enumerate(colors):
        ax.scatter(
            X[y == i, 2],
            X[y == i, 3],
            label=species[i],
            alpha=0.6,
            c=color
        )

    ax.scatter(
        petal_length,
        petal_width,
        color="red",
        s=50,
        marker="X",
        label="あなたの入力"
    )

    ax.set_xlabel("花弁の長さ")
    ax.set_ylabel("花弁の幅")
    ax.set_title("アヤメデータ散布図と入力プロット")
    ax.legend()

    st.pyplot(fig)
