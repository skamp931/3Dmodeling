import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Advanced 3D Viewer")

# --- セッション状態の初期化 ---
# st.session_state を使うと、ウィジェットを操作しても変数の値が保持されます。
def init_session_state():
    if 'lines' not in st.session_state:
        st.session_state.lines = []
    # 物体の初期Z座標オフセット（敷地に埋まるように調整）
    if 'object_z_offset' not in st.session_state:
        st.session_state.object_z_offset = -2.5
    # 計測結果を保持
    if 'measurement' not in st.session_state:
        st.session_state.measurement = None
    # 埋設体積を保持
    if 'buried_volume' not in st.session_state:
        st.session_state.buried_volume = None

init_session_state()

# --- 定数と平面関数の定義 ---
# 敷地となる平面の方程式 z = ax + by + c
PLANE_EQ = {'a': 0.1, 'b': -0.05, 'c': 0}

def get_plane_z(x, y):
    """指定されたx, y座標に対する平面の高さを返す"""
    return PLANE_EQ['a'] * x + PLANE_EQ['b'] * y + PLANE_EQ['c']

# --- 3Dデータを作成する関数 ---

def create_site_mesh():
    """斜めの敷地のメッシュデータを作成"""
    points_2d = np.random.uniform(-15, 15, size=(200, 2))
    tri = Delaunay(points_2d)
    z = get_plane_z(points_2d[:, 0], points_2d[:, 1])
    site_points_3d = np.c_[points_2d[:, 0], points_2d[:, 1], z]
    return site_points_3d, tri.simplices

def create_object_mesh(z_offset=0.0):
    """物体のメッシュデータ（立方体）を作成"""
    # === エラー修正箇所 ===
    # 頂点座標を整数(int)から小数(float)で定義します。例: -1 -> -1.0
    base_vertices = np.array([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]
    ])
    # サイズを2倍にし、指定されたオフセットでZ位置を調整
    vertices = base_vertices * 2
    # これで float型の配列に float型の値を加算できるようになり、エラーが解消されます
    vertices[:, 2] += z_offset
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], 
        [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], 
        [3, 0, 4], [3, 4, 7]
    ])
    return vertices, faces

def calculate_buried_volume(object_vertices, plane_func, samples=50000):
    """モンテカルロ法で埋設体積を概算する"""
    min_coords = object_vertices.min(axis=0)
    max_coords = object_vertices.max(axis=0)
    
    dims = max_coords - min_coords
    total_volume = dims[0] * dims[1] * dims[2]
    if total_volume == 0:
        return 0

    # 物体のバウンディングボックス内にランダムな点を生成
    random_points = np.random.rand(samples, 3) * dims + min_coords
    
    # 点が平面の下にあるかどうかをチェック
    plane_z_at_points = plane_func(random_points[:, 0], random_points[:, 1])
    buried_mask = random_points[:, 2] < plane_z_at_points
    
    # 埋まっている点の割合から体積を計算
    buried_ratio = np.sum(buried_mask) / samples
    return total_volume * buried_ratio

# --- メインのアプリケーション部分 ---
st.title("高機能3Dビューア")

# --- サイドバーのUI ---
st.sidebar.title("🛠️ ツール")

# セクション1: 物体の操作と体積計算
with st.sidebar.expander("📦 物体コントロール", expanded=True):
    st.write("ボタンで物体を上下に移動できます。")
    col1, col2, col3 = st.columns([1,1,1.5])
    if col1.button("⬆️ 上へ"):
        st.session_state.object_z_offset += 0.5
        st.session_state.buried_volume = None  # 移動したら体積をリセット
    if col2.button("⬇️ 下へ"):
        st.session_state.object_z_offset -= 0.5
        st.session_state.buried_volume = None

    if col3.button("🔄 位置リセット"):
        st.session_state.object_z_offset = -2.5
        st.session_state.buried_volume = None

    st.write("---")
    if st.button("埋設体積を計算", key="calc_vol"):
        verts, _ = create_object_mesh(st.session_state.object_z_offset)
        volume = calculate_buried_volume(verts, get_plane_z)
        st.session_state.buried_volume = volume
    
    if st.session_state.buried_volume is not None:
        st.metric("埋設部分の体積 (概算)", f"{st.session_state.buried_volume:.2f} m³")


# セクション2: 線の追加
with st.sidebar.expander("📏 線を追加", expanded=False):
    st.write("始点と終点の座標を入力して線を追加します。")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("X (始点)", -20.0, 20.0, 0.0, 1.0, key="x1")
        y1 = st.number_input("Y (始点)", -20.0, 20.0, -8.0, 1.0, key="y1")
        z1 = st.number_input("Z (始点)", -20.0, 20.0, 5.0, 1.0, key="z1")
    with col2:
        x2 = st.number_input("X (終点)", -20.0, 20.0, 8.0, 1.0, key="x2")
        y2 = st.number_input("Y (終点)", -20.0, 20.0, -8.0, 1.0, key="y2")
        z2 = st.number_input("Z (終点)", -20.0, 20.0, 0.0, 1.0, key="z2")

    if st.button("線を追加"):
        st.session_state.lines.append({"start": [x1, y1, z1], "end": [x2, y2, z2]})
    if st.button("全ての線を削除"):
        st.session_state.lines = []


# セクション3: 距離の計測
with st.sidebar.expander("📐 2点間の距離を計測", expanded=False):
    col3, col4 = st.columns(2)
    with col3:
        xa = st.number_input("X (点1)", -20.0, 20.0, -5.0, 1.0, key="xa")
        ya = st.number_input("Y (点1)", -20.0, 20.0, -5.0, 1.0, key="ya")
        za = st.number_input("Z (点1)", -20.0, 20.0, 0.0, 1.0, key="za")
    with col4:
        xb = st.number_input("X (点2)", -20.0, 20.0, 5.0, 1.0, key="xb")
        yb = st.number_input("Y (点2)", -20.0, 20.0, 5.0, 1.0, key="yb")
        zb = st.number_input("Z (点2)", -20.0, 20.0, 4.0, 1.0, key="zb")

    if st.button("距離を計算"):
        p1 = np.array([xa, ya, za]); p2 = np.array([xb, yb, zb])
        dist = np.linalg.norm(p1 - p2)
        st.session_state.measurement = {"p1": p1, "p2": p2, "dist": dist}
    
    if st.session_state.measurement:
        st.metric("計測距離", f"{st.session_state.measurement['dist']:.2f}")


# --- 3Dグラフの描画 ---
fig = go.Figure()

# データを作成
site_vertices, site_faces = create_site_mesh()
object_vertices, object_faces = create_object_mesh(st.session_state.object_z_offset)

# 1. 敷地
fig.add_trace(go.Mesh3d(x=site_vertices[:,0], y=site_vertices[:,1], z=site_vertices[:,2],
    i=site_faces[:,0], j=site_faces[:,1], k=site_faces[:,2],
    color='lightgreen', opacity=0.7, name='敷地', hoverinfo='none'))

# 2. 物体
fig.add_trace(go.Mesh3d(x=object_vertices[:,0], y=object_vertices[:,1], z=object_vertices[:,2],
    i=object_faces[:,0], j=object_faces[:,1], k=object_faces[:,2],
    color='royalblue', opacity=1.0, name='物体', hoverinfo='none'))

# 3. 追加された線と座標を表示
for i, line in enumerate(st.session_state.lines):
    start, end = line["start"], line["end"]
    fig.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode='lines', line=dict(color='red', width=5), name=f'追加線{i+1}'))
    # 座標ラベルを追加
    fig.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode='text', text=[f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})" for p in [start, end]],
        textfont=dict(color="darkred", size=10), textposition='middle right', hoverinfo='none'))

# 4. 計測した距離を表示
if st.session_state.measurement:
    m = st.session_state.measurement
    p1, p2, dist = m['p1'], m['p2'], m['dist']
    mid_point = (p1 + p2) / 2
    # 計測線
    fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode='lines', line=dict(color='orange', width=7, dash='dash'), name='計測線'))
    # 距離ラベル
    fig.add_trace(go.Scatter3d(x=[mid_point[0]], y=[mid_point[1]], z=[mid_point[2]],
        mode='text', text=[f"距離: {dist:.2f}"],
        textfont=dict(color="orange", size=12), hoverinfo='none'))


# グラフのレイアウト設定
fig.update_layout(
    title_text='インタラクティブ3Dビューア',
    scene=dict(
        xaxis=dict(title='X軸', range=[-15, 15]),
        yaxis=dict(title='Y軸', range=[-15, 15]),
        zaxis=dict(title='Z軸 (高さ)', range=[-10, 15]),
        aspectratio=dict(x=1, y=1, z=0.5),
        camera_eye=dict(x=1.8, y=1.8, z=1.2)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True, height=700)
