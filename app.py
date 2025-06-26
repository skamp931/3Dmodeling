import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay
from streamlit_plotly_events import plotly_events
import pandas as pd

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Site and Pillar Viewer")

# --- セッション状態の初期化 ---
def init_session_state():
    if 'pillar_offsets' not in st.session_state:
        st.session_state.pillar_offsets = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    # 距離計測用
    if 'measurement' not in st.session_state:
        st.session_state.measurement = None
    if 'measure_mode' not in st.session_state:
        st.session_state.measure_mode = False
    if 'measure_points' not in st.session_state:
        st.session_state.measure_points = []
    # 描画用
    if 'lines' not in st.session_state:
        st.session_state.lines = []
    if 'drawing_points' not in st.session_state:
        st.session_state.drawing_points = []
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = False


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
            st.rerun()
        if st.button(f"⬇️##{pillar_id}", use_container_width=True):
            st.session_state.pillar_offsets[pillar_id] -= 0.5
            st.rerun()

st.divider()

# --- 2点間距離の計測パネル ---
st.subheader("２点間距離の計測（クリック操作）")
measure_cols = st.columns([1, 3])
with measure_cols[0]:
    st.session_state.measure_mode = st.toggle("計測モードを有効にする", value=st.session_state.measure_mode, key="measure_toggle")
    if st.session_state.measure_mode:
        if st.session_state.measure_points:
            st.info("終点をクリックしてください。")
        else:
            st.info("始点をクリックしてください。")

with measure_cols[1]:
    if st.session_state.measurement:
        st.metric("計測距離", f"{st.session_state.measurement['dist']:.2f} m")

st.divider()

# --- サイドバー ---
st.sidebar.title("🛠️ ツール")
with st.sidebar.expander("✏️ 線を描画", expanded=True):
    st.session_state.drawing_mode = st.toggle("描画モードを有効にする", value=st.session_state.drawing_mode, key="draw_toggle")
    if st.session_state.drawing_mode:
        if st.session_state.drawing_points:
            st.info("終点をクリックしてください。")
        else:
            st.info("始点をクリックしてください。")

    if st.button("全ての線と計測をクリア"):
        st.session_state.lines = []
        st.session_state.drawing_points = []
        st.session_state.measurement = None
        st.session_state.measure_points = []
        st.rerun()


# --- 3Dグラフ描画 ---
fig = go.Figure()

# 敷地
if site_vertices is not None and site_vertices.size > 0:
    verts, faces = create_mesh_from_vertices(site_vertices)
    if verts.size > 0 and faces.size > 0:
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='burlywood', opacity=0.8, name="Site"))

# 柱
if pillars_config:
    for pillar_id, config in pillars_config.items():
        x, y = config['pos']; z_off = st.session_state.pillar_offsets[pillar_id]; total_h = config['base_cyl_h'] + config['main_cyl_h']; init_z = get_plane_z(x, y) - (total_h * 4/5)
        base_pos = [x, y, init_z + z_off]; main_pos = [base_pos[0], base_pos[1], base_pos[2] + config['base_cyl_h']]
        verts, faces = create_cylinder_mesh(base_pos, config['base_cyl_r'], config['base_cyl_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='darkgrey'))
        verts, faces = create_cylinder_mesh(main_pos, config['main_cyl_r'], config['main_cyl_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='lightslategray'))
        line_start = [main_pos[0], main_pos[1], main_pos[2] + config['main_cyl_h']]; line_end = [line_start[0], line_start[1], line_start[2] + 1.5]; fig.add_trace(go.Scatter3d(x=[line_start[0],line_end[0]],y=[line_start[1],line_end[1]],z=[line_start[2],line_end[2]],mode='lines',line=dict(color='red',width=7)))

# 計測線の描画
if st.session_state.measurement:
    m = st.session_state.measurement; p1, p2, dist = m['p1'], m['p2'], m['dist']; mid_point = (p1 + p2) / 2
    fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color='orange', width=7, dash='dash')))
    fig.add_trace(go.Scatter3d(x=[mid_point[0]], y=[mid_point[1]], z=[mid_point[2]], mode='text', text=[f"距離: {dist:.2f}"], textfont=dict(color="orange", size=12), hoverinfo='none'))

# 描画中の線やマーカー
if st.session_state.drawing_points:
    pt = st.session_state.drawing_points[0]
    fig.add_trace(go.Scatter3d(x=[pt['x']], y=[pt['y']], z=[pt['z']], mode='markers', marker=dict(color='cyan', size=10, symbol='cross')))
if st.session_state.measure_points:
    pt = st.session_state.measure_points[0]
    fig.add_trace(go.Scatter3d(x=[pt['x']], y=[pt['y']], z=[pt['z']], mode='markers', marker=dict(color='magenta', size=10, symbol='cross')))
for line in st.session_state.lines:
    fig.add_trace(go.Scatter3d(x=[line["start"]['x'],line["end"]['x']],y=[line["start"]['y'],line["end"]['y']],z=[line["start"]['z'],line["end"]['z']],mode='lines',line=dict(color='cyan',width=5)))

# レイアウト設定
fig.update_layout(scene=dict(xaxis=dict(title='X (m)',range=[-10,10]),yaxis=dict(title='Y (m)',range=[-10,10]),zaxis=dict(title='Z (m)',range=[-10,10]),aspectratio=dict(x=1,y=1,z=1)), margin=dict(l=0,r=0,b=0,t=40), showlegend=False)

# --- イベント処理 ---
selected_points = plotly_events(fig, click_event=True, key="plotly_events")

if selected_points:
    clicked_point = selected_points[0]
    # 計測モードの処理
    if st.session_state.measure_mode:
        if not st.session_state.measure_points:
            st.session_state.measure_points.append(clicked_point)
            st.rerun()
        else:
            p1_dict = st.session_state.measure_points.pop(0)
            p2_dict = clicked_point
            p1 = np.array([p1_dict['x'], p1_dict['y'], p1_dict['z']])
            p2 = np.array([p2_dict['x'], p2_dict['y'], p2_dict['z']])
            dist = np.linalg.norm(p1 - p2)
            st.session_state.measurement = {"p1": p1, "p2": p2, "dist": dist}
            st.session_state.measure_mode = False # 計測が終わったらモードをオフに
            st.rerun()
    # 描画モードの処理
    elif st.session_state.drawing_mode:
        if not st.session_state.drawing_points:
            st.session_state.drawing_points.append(clicked_point)
            st.rerun()
        else:
            start_point = st.session_state.drawing_points.pop(0)
            st.session_state.lines.append({"start": start_point, "end": clicked_point})
            st.rerun()
