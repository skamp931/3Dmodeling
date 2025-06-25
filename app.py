import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 3D表示のためのサンプルデータを作成 ---

# 1. 敷地データ (XYZの点群)
#    -10から10の範囲に、200個の点をランダムに生成
site_points = np.random.rand(200, 3) * 20 - 10
# Z座標（高さ）を低くして地面のように見せる
site_points[:, 2] = np.random.uniform(-1, 0, 200)

# 2. 物体データ (立方体の形状)
#    立方体の8つの頂点座標を定義
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面の4点
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 上面の4点
])
# 中心に移動させ、サイズを調整
vertices = (vertices - 0.5) * 4
# Z軸方向に少し持ち上げる
vertices[:, 2] += 2

# 立方体の12個の三角形の面を定義 (2つの三角形で1つの四角い面を作る)
faces = np.array([
    [0, 1, 2], [0, 2, 3],  # 底面
    [4, 5, 6], [4, 6, 7],  # 上面
    [0, 1, 5], [0, 5, 4],  # 側面1
    [1, 2, 6], [1, 6, 5],  # 側面2
    [2, 3, 7], [2, 7, 6],  # 側面3
    [3, 0, 4], [3, 4, 7]   # 側面4
])

# --- Streamlit アプリの作成 ---

st.set_page_config(layout="wide")
st.title("敷地と物体の3D表示")

st.write("敷地の点群データと、中央に配置された立方体を3Dで表示します。")

# --- 3Dグラフの作成 ---

# グラフオブジェクトを初期化
fig = go.Figure()

# 1. 敷地データを3D散布図として追加
fig.add_trace(go.Scatter3d(
    x=site_points[:, 0],
    y=site_points[:, 1],
    z=site_points[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color='green',
        opacity=0.6
    ),
    name='敷地'
))

# 2. 物体データを3Dメッシュとして追加
fig.add_trace(go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    color='blue',
    opacity=0.7,
    name='物体'
))

# グラフのレイアウト設定
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X軸', range=[-12, 12]),
        yaxis=dict(title='Y軸', range=[-12, 12]),
        zaxis=dict(title='Z軸 (高さ)', range=[-2, 10]),
        # 各軸のスケールを合わせる
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Streamlitでグラフを表示
st.plotly_chart(fig, use_container_width=True)

st.info("""
**操作方法:**
- **ドラッグ:** 視点を回転できます。
- **マウスホイール:** ズームイン・ズームアウトできます。
""")
