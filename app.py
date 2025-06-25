import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="3D Viewer")

# --- セッション状態の初期化 ---
# st.session_state を使うと、ウィジェットを操作しても変数の値が保持されます。
# 'lines' というキーがなければ、空のリストで初期化します。
if 'lines' not in st.session_state:
    st.session_state.lines = []


# --- 3D表示のためのサンプルデータを作成する関数 ---

def create_site_mesh():
    """敷地のメッシュデータを作成"""
    # XY平面上にランダムな点を生成
    points_2d = np.random.rand(150, 2) * 24 - 12
    
    # Delaunay三角分割を使って、点の集まりから三角形のメッシュを生成
    # これにより、どの3つの点を結べば面になるかが決まる (tri.simplices)
    tri = Delaunay(points_2d)

    # Z座標（高さ）にランダムな凹凸を与える
    z = np.random.uniform(-1.5, -0.5, size=points_2d.shape[0])
    
    # 2Dの点群にZ座標を結合して3Dの点群にする
    site_points_3d = np.c_[points_2d[:, 0], points_2d[:, 1], z]

    return site_points_3d, tri.simplices

def create_object_mesh():
    """物体のメッシュデータ（立方体）を作成"""
    # 立方体の8つの頂点座標を定義
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    # 中心に移動させ、サイズを調整し、少し持ち上げる
    vertices = (vertices - 0.5) * 4
    vertices[:, 2] += 2

    # 立方体の12個の三角形の面を定義
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
    ])
    return vertices, faces

# --- メインのアプリケーション部分 ---

st.title("敷地と物体の3Dビューア")

# --- サイドバーのUI ---

st.sidebar.title("ツール")

# セクション1: 線の追加
with st.sidebar.expander("線を追加", expanded=True):
    st.write("始点と終点の座標を入力して線を追加します。")
    # 2つのカラムを作成してUIを整理
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("始点 (A)")
        x1 = st.number_input("X座標 (A)", -20.0, 20.0, 0.0, 1.0, key="x1")
        y1 = st.number_input("Y座標 (A)", -20.0, 20.0, -5.0, 1.0, key="y1")
        z1 = st.number_input("Z座標 (A)", -20.0, 20.0, 5.0, 1.0, key="z1")
    with col2:
        st.subheader("終点 (B)")
        x2 = st.number_input("X座標 (B)", -20.0, 20.0, 5.0, 1.0, key="x2")
        y2 = st.number_input("Y座標 (B)", -20.0, 20.0, 5.0, 1.0, key="y2")
        z2 = st.number_input("Z座標 (B)", -20.0, 20.0, 0.0, 1.0, key="z2")

    # ボタンが押されたら、入力された座標をリストに追加
    if st.button("線を追加"):
        new_line = {
            "start": [x1, y1, z1],
            "end": [x2, y2, z2]
        }
        st.session_state.lines.append(new_line)

    # 追加した線をクリアするボタン
    if st.button("全ての線を削除"):
        st.session_state.lines = []


# セクション2: 距離の計測
with st.sidebar.expander("2点間の距離を計測", expanded=True):
    st.write("2点の座標を入力して距離を計算します。")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("点1")
        xa = st.number_input("X座標 (点1)", -20.0, 20.0, -3.0, 1.0, key="xa")
        ya = st.number_input("Y座標 (点1)", -20.0, 20.0, -3.0, 1.0, key="ya")
        za = st.number_input("Z座標 (点1)", -20.0, 20.0, 0.0, 1.0, key="za")
    with col4:
        st.subheader("点2")
        xb = st.number_input("X座標 (点2)", -20.0, 20.0, 3.0, 1.0, key="xb")
        yb = st.number_input("Y座標 (点2)", -20.0, 20.0, 3.0, 1.0, key="yb")
        zb = st.number_input("Z座標 (点2)", -20.0, 20.0, 4.0, 1.0, key="zb")

    # ボタンが押されたら距離を計算して表示
    if st.button("距離を計算"):
        point_a = np.array([xa, ya, za])
        point_b = np.array([xb, yb, zb])
        # np.linalg.normで2点間のユークリッド距離を計算
        distance = np.linalg.norm(point_a - point_b)
        st.metric("2点間の距離", f"{distance:.2f}")


# --- 3Dグラフの描画 ---

# グラフオブジェクトを初期化
fig = go.Figure()

# データを作成
site_vertices, site_faces = create_site_mesh()
object_vertices, object_faces = create_object_mesh()

# 1. 敷地データを3Dメッシュとして追加
fig.add_trace(go.Mesh3d(
    x=site_vertices[:, 0],
    y=site_vertices[:, 1],
    z=site_vertices[:, 2],
    i=site_faces[:, 0],
    j=site_faces[:, 1],
    k=site_faces[:, 2],
    color='lightgreen',
    opacity=0.8,
    name='敷地',
    hoverinfo='none' # ホバーしても情報を表示しない
))

# 2. 物体データを3Dメッシュとして追加
fig.add_trace(go.Mesh3d(
    x=object_vertices[:, 0],
    y=object_vertices[:, 1],
    z=object_vertices[:, 2],
    i=object_faces[:, 0],
    j=object_faces[:, 1],
    k=object_faces[:, 2],
    color='royalblue',
    opacity=0.9,
    name='物体',
    hoverinfo='none'
))

# 3. 追加された線をグラフに描画
for i, line in enumerate(st.session_state.lines):
    start = line["start"]
    end = line["end"]
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color='red', width=5),
        name=f'追加した線 {i+1}'
    ))


# グラフのレイアウト設定
fig.update_layout(
    title_text='3Dビューア',
    scene=dict(
        xaxis=dict(title='X軸', range=[-15, 15]),
        yaxis=dict(title='Y軸', range=[-15, 15]),
        zaxis=dict(title='Z軸 (高さ)', range=[-5, 15]),
        aspectratio=dict(x=1, y=1, z=0.5), # 各軸のスケールの比率
        camera_eye=dict(x=1.5, y=1.5, z=1.0) # 初期カメラ視点
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Streamlitでグラフを表示
st.plotly_chart(fig, use_container_width=True, height=700)

st.info("""
**操作方法:**
- **ドラッグ:** 視点を回転できます。
- **マウスホイール:** ズームイン・ズームアウトできます。
- **左のサイドバー:** 線を追加したり、2点間の距離を計測したりできます。
""")
