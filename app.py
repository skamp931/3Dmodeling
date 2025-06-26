import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay
from streamlit_plotly_events import plotly_events # この行は将来のために残します
import pandas as pd

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Site and Pillar Viewer")

# --- セッション状態の初期化 ---
def init_session_state():
    # 柱の上下移動オフセットを管理
    if 'pillar_offsets' not in st.session_state:
        st.session_state.pillar_offsets = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    # 距離計測の結果を管理
    if 'measurement' not in st.session_state:
        st.session_state.measurement = None

# --- 3Dデータを作成する関数 ---
def get_plane_z(x, y, slope_degrees=30):
    """指定された傾斜角を持つ敷地平面の高さを返す"""
    slope_rad = np.deg2rad(slope_degrees)
    return y * np.tan(slope_rad)

def create_mesh_from_vertices(vertices):
    """頂点群からDelaunay三角分割でメッシュを作成する"""
    try:
        if vertices is None or vertices.shape[0] < 3:
            return np.array([]), np.array([])
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        return vertices, tri.simplices
    except Exception as e:
        st.error(f"メッシュの生成に失敗しました: {e}")
        return np.array([]), np.array([])

def create_cylinder_mesh(center_pos, radius, height, n_segments=32):
    """円柱の頂点と面データを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x_c, y_c = radius * np.cos(theta), radius * np.sin(theta)
    
    verts = []
    # Bottom, Top, Center points
    for i in range(n_segments): verts.append([x_c[i], y_c[i], 0])
    for i in range(n_segments): verts.append([x_c[i], y_c[i], height])
    verts.append([0, 0, 0])      # Bottom center (index: 2n)
    verts.append([0, 0, height]) # Top center (index: 2n+1)
    
    verts = np.array(verts, dtype=float) + np.array(center_pos, dtype=float)
    
    faces = []
    # Sides
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.extend([[i, next_i, i + n_segments], [next_i, i + n_segments, next_i + n_segments]])
    # Caps
    for i in range(n_segments):
        faces.append([i, (i + 1) % n_segments, 2 * n_segments])
        faces.append([i + n_segments, ((i + 1) % n_segments) + n_segments, 2 * n_segments + 1])
        
    return verts, np.array(faces, dtype=int)


# --- データ定義 ---
def get_default_site_data():
    """デフォルトの敷地データを生成する"""
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y)
    z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

def get_default_pillars_config():
    """デフォルトの柱データを生成する"""
    dist = 7.0 / 2.0
    config = {
        'A': {'pos': [-dist, dist], 'base_cyl_r': 2.5, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
        'B': {'pos': [dist, dist], 'base_cyl_r': 2.5, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
        'C': {'pos': [dist, -dist], 'base_cyl_r': 2.5, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
        'D': {'pos': [-dist, -dist], 'base_cyl_r': 2.5, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
    }
    return config

# --- メインアプリケーション ---
init_session_state()
st.title("敷地と柱の3Dビューア")

# データを生成
site_vertices = get_default_site_data()
pillars_config = get_default_pillars_config()

# --- 操作パネル ---
st.subheader("各脚の操作")
cols = st.columns(len(pillars_config))
for col, pillar_id in zip(cols, pillars_config.keys()):
    with col:
        st.markdown(f"**{pillar_id}脚**")
        if st.button(f"⬆️##{pillar_id}", use_container_width=True):
            st.session_state.pillar_offsets[pillar_id] += 0.5
        if st.button(f"⬇️##{pillar_id}", use_container_width=True):
            st.session_state.pillar_offsets[pillar_id] -= 0.5

st.divider()

# --- 2点間距離の計測パネル ---
st.subheader("２点間距離の計測")
m_cols = st.columns([2, 2, 1, 2])
with m_cols[0]:
    st.markdown("**点1**")
    xa = st.number_input("X座標 (1)", -20.0, 20.0, -5.0, 1.0, key="xa", label_visibility="collapsed")
    ya = st.number_input("Y座標 (1)", -20.0, 20.0, -5.0, 1.0, key="ya", label_visibility="collapsed")
    za = st.number_input("Z座標 (1)", -20.0, 20.0, 0.0, 1.0, key="za", label_visibility="collapsed")

with m_cols[1]:
    st.markdown("**点2**")
    xb = st.number_input("X座標 (2)", -20.0, 20.0, 5.0, 1.0, key="xb", label_visibility="collapsed")
    yb = st.number_input("Y座標 (2)", -20.0, 20.0, 5.0, 1.0, key="yb", label_visibility="collapsed")
    zb = st.number_input("Z座標 (2)", -20.0, 20.0, 4.0, 1.0, key="zb", label_visibility="collapsed")

with m_cols[2]:
    st.markdown("　") # スペース調整
    if st.button("距離を計算", use_container_width=True):
        p1 = np.array([xa, ya, za])
        p2 = np.array([xb, yb, zb])
        dist = np.linalg.norm(p1 - p2)
        st.session_state.measurement = {"p1": p1, "p2": p2, "dist": dist}

with m_cols[3]:
    if st.session_state.measurement:
        st.metric("計測距離", f"{st.session_state.measurement['dist']:.2f} m")

st.divider()

# --- 3Dグラフ描画 ---
fig = go.Figure()

# 敷地
if site_vertices is not None and site_vertices.size > 0:
    verts, faces = create_mesh_from_vertices(site_vertices)
    if verts.size > 0 and faces.size > 0:
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='burlywood', opacity=0.8, name="Site"
        ))

# 柱
if pillars_config:
    for pillar_id, config in pillars_config.items():
        x, y = config['pos']
        z_off = st.session_state.pillar_offsets[pillar_id]
        total_h = config['base_cyl_h'] + config['main_cyl_h']
        init_z = get_plane_z(x, y) - (total_h * 4/5)
        
        base_pos = [x, y, init_z + z_off]
        main_pos = [base_pos[0], base_pos[1], base_pos[2] + config['base_cyl_h']]
        
        verts, faces = create_cylinder_mesh(base_pos, config['base_cyl_r'], config['base_cyl_h'])
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='darkgrey'))
        
        verts, faces = create_cylinder_mesh(main_pos, config['main_cyl_r'], config['main_cyl_h'])
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='lightslategray'))
        
        line_start = [main_pos[0], main_pos[1], main_pos[2] + config['main_cyl_h']]
        line_end = [line_start[0], line_start[1], line_start[2] + 1.5]
        fig.add_trace(go.Scatter3d(x=[line_start[0],line_end[0]],y=[line_start[1],line_end[1]],z=[line_start[2],line_end[2]],mode='lines',line=dict(color='red',width=7)))

# 計測線の描画
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

# レイアウト設定
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (m)', range=[-10, 10]),
        yaxis=dict(title='Y (m)', range=[-10, 10]),
        zaxis=dict(title='Z (m)', range=[-10, 10]),
        aspectratio=dict(x=1, y=1, z=1)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False
)

# Streamlitで表示
st.plotly_chart(fig, use_container_width=True)
