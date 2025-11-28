import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_iris
import os

# 日本語フォント設定 (Streamlit Cloud用)
# fontフォルダがルートにあることを想定
try:
    matplotlib.font_manager.fontManager.addfont("font/NotoSansJP-Regular.ttf")
    plt.rcParams['font.family'] = "Noto Sans JP"
except FileNotFoundError:
    st.warning("日本語フォント (font/NotoSansJP-Regular.ttf) が見つかりません。デフォルトフォントを使用します。")

# モデルの読み込み
# 実行ディレクトリによってパスが変わる可能性があるため、両方試す
model_path = 'src/model.pkl'
if not os.path.exists(model_path):
    model_path = 'model.pkl'

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("モデルファイル (model.pkl) が見つかりません。train_model.py を実行してモデルを作成してください。")
    st.stop()

# タイトル
st.title("Iris 品種予測アプリ")

# 入力フォーム
st.write("以下の数値を入力してください。")
sepal_length = st.number_input("がく片の長さ", 0.0, 10.0, 5.0)
sepal_width = st.number_input("がく片の幅", 0.0, 10.0, 3.5)
petal_length = st.number_input("花弁の長さ", 0.0, 10.0, 1.4)
petal_width = st.number_input("花弁の幅", 0.0, 10.0, 0.2)

species = ["setosa", "versicolor", "virginica"]

if st.button("予測"):
    # 予測の実行
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    st.success(f"予測結果: {species[prediction]}")

    # 散布図の表示
    st.subheader("散布図 (花弁の長さ vs 花弁の幅)")

    iris = load_iris()
    X = iris.data
    y = iris.target

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["blue", "orange", "green"]

    # 学習データのプロット (花弁の長さ(index 2) と 花弁の幅(index 3) を使用)
    for i, color in enumerate(colors):
        ax.scatter(X[y == i, 2], X[y == i, 3], alpha=0.6, c=color, label=species[i])

    # 入力データのプロット
    ax.scatter(petal_length, petal_width, color="red", marker="X", s=100, label="あなたの入力")

    ax.set_xlabel("花弁の長さ")
    ax.set_ylabel("花弁の幅")
    ax.set_title("Irisデータ散布図と入力プロット")
    ax.legend()

    st.pyplot(fig)
