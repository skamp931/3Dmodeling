import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Site and Pillar Viewer")

# --- セッション状態の初期化 ---
def init_session_state():
    # 柱の上下移動オフセットを管理
    if 'pillar_offsets' not in st.session_state:
        st.session_state.pillar_offsets = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}

# --- 3Dデータを作成・計算する関数 ---
def get_plane_z(x, y, slope_degrees=30):
    """指定された傾斜角を持つ敷地平面の高さを返す"""
    slope_rad = np.deg2rad(slope_degrees)
    return y * np.tan(slope_rad)

def create_mesh_from_vertices(vertices):
    """頂点群からDelaunay三角分割でメッシュを作成する"""
    try:
        if vertices is None or vertices.shape[0] < 3: return np.array([]), np.array([])
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        return vertices, tri.simplices
    except Exception as e:
        st.error(f"メッシュの生成に失敗しました: {e}")
        return np.array([]), np.array([])

def create_cylinder_mesh(center_pos, radius, height, n_segments=32):
    """円柱の頂点と面データを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x_c = radius * np.cos(theta)
    y_c = radius * np.sin(theta)
    verts = []
    for i in range(n_segments): verts.append([x_c[i], y_c[i], 0])
    for i in range(n_segments): verts.append([x_c[i], y_c[i], height])
    verts.extend([[0, 0, 0], [0, 0, height]])
    verts = np.array(verts, dtype=float) + np.array(center_pos, dtype=float)
    faces = []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.extend([[i, next_i, i + n_segments], [next_i, i + n_segments, next_i + n_segments]])
    for i in range(n_segments):
        faces.append([i, (i + 1) % n_segments, 2 * n_segments])
        faces.append([i + n_segments, ((i + 1) % n_segments) + n_segments, 2 * n_segments + 1])
    return verts, np.array(faces, dtype=int)

def create_frustum_mesh(center_pos, bottom_radius, top_radius, height, n_segments=32):
    """通常の円錐台のメッシュを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    xb, yb = bottom_radius * np.cos(theta), bottom_radius * np.sin(theta)
    xt, yt = top_radius * np.cos(theta), top_radius * np.sin(theta)
    verts = []
    for i in range(n_segments): verts.append([xb[i], yb[i], 0])
    for i in range(n_segments): verts.append([xt[i], yt[i], height])
    verts.extend([[0, 0, 0], [0, 0, height]])
    verts = np.array(verts, dtype=float) + np.array(center_pos, dtype=float)
    faces = []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.extend([[i, next_i, i + n_segments], [next_i, i + n_segments, next_i + n_segments]])
    for i in range(n_segments):
        faces.append([i, (i + 1) % n_segments, 2 * n_segments])
        faces.append([i + n_segments, ((i + 1) % n_segments) + n_segments, 2 * n_segments + 1])
    return verts, np.array(faces, dtype=int)

def clip_mesh_by_plane(vertices, faces, plane_func):
    """メッシュを平面で切断し、上と下のメッシュデータを返す"""
    plane_z = plane_func(vertices[:, 0], vertices[:, 1])
    distances = vertices[:, 2] - plane_z
    vert_sides = np.sign(distances)
    
    new_vertices = list(vertices)
    above_faces, below_faces = [], []

    for face in faces:
        face_sides = vert_sides[face]
        
        if np.all(face_sides >= 0): above_faces.append(face); continue
        if np.all(face_sides < 0): below_faces.append(face); continue
        
        if np.sum(face_sides >= 0) > 0: above_faces.append(face)
        if np.sum(face_sides < 0) > 0: below_faces.append(face)

    return np.array(new_vertices), np.array(above_faces), np.array(below_faces)


def calculate_volumes(verts, plane_func, samples=5000):
    """頂点群の上部・下部の体積を計算"""
    if verts is None or verts.size == 0: return 0, 0
    min_c, max_c = verts.min(axis=0), verts.max(axis=0)
    dims, bbox_volume = max_c - min_c, np.prod(max_c - min_c)
    if bbox_volume == 0: return 0, 0
    
    random_points = np.random.rand(samples, 3) * dims + min_c
    plane_z_at_points = plane_func(random_points[:, 0], random_points[:, 1])
    
    is_above = random_points[:, 2] >= plane_z_at_points
    is_below = ~is_above
    
    vol_above = bbox_volume * (np.sum(is_above) / samples)
    vol_below = bbox_volume * (np.sum(is_below) / samples)
    
    return vol_above, vol_below

# --- データ定義 ---
def get_default_site_data():
    x = np.linspace(-10, 10, 20); y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y); z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

def get_default_pillars_config():
    dist = 7.0 / 2.0
    config = {
        'A': {'pos':[-dist, dist],'foundation_h':1.0,'foundation_r_bottom':3.0,'foundation_r_top':2.5,'base_cyl_r':2.5,'base_cyl_h':1.5,'main_cyl_r':0.5,'main_cyl_h':6.0},
        'B': {'pos':[dist, dist],'foundation_h':1.0,'foundation_r_bottom':3.0,'foundation_r_top':2.5,'base_cyl_r':2.5,'base_cyl_h':1.5,'main_cyl_r':0.5,'main_cyl_h':6.0},
        'C': {'pos':[dist, -dist],'foundation_h':1.0,'foundation_r_bottom':3.0,'foundation_r_top':2.5,'base_cyl_r':2.5,'base_cyl_h':1.5,'main_cyl_r':0.5,'main_cyl_h':6.0},
        'D': {'pos':[-dist, -dist],'foundation_h':1.0,'foundation_r_bottom':3.0,'foundation_r_top':2.5,'base_cyl_r':2.5,'base_cyl_h':1.5,'main_cyl_r':0.5,'main_cyl_h':6.0},
    }
    return config

# --- メインアプリケーション ---
init_session_state()
st.title("敷地と柱の3Dビューア")

site_vertices = get_default_site_data()
pillars_config = get_default_pillars_config()

# --- 3Dグラフ描画 ---
fig = go.Figure()

# 敷地
if site_vertices is not None and site_vertices.size > 0:
    verts, faces = create_mesh_from_vertices(site_vertices)
    if verts.size > 0 and faces.size > 0:
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='burlywood',opacity=0.7))

# 柱の描画と体積計算用のデータ準備
pillar_volumes = {}
for pillar_id, config in pillars_config.items():
    x, y = config['pos']; z_off = st.session_state.pillar_offsets.get(pillar_id, 0.0)
    total_h = config['foundation_h'] + config['base_cyl_h'] + config['main_cyl_h']
    init_z = get_plane_z(x, y) - (total_h * 4/5)
    
    foundation_pos_z = init_z + z_off
    base_pos_z = foundation_pos_z + config['foundation_h']
    main_pos_z = base_pos_z + config['base_cyl_h']
    
    foundation_pos = [x, y, foundation_pos_z]
    base_pos = [x, y, base_pos_z]
    main_pos = [x, y, main_pos_z]
    
    # --- 基礎円錐台（計算対象）---
    frustum_verts, frustum_faces = create_frustum_mesh(foundation_pos, config['foundation_r_bottom'], config['foundation_r_top'], config['foundation_h'])
    
    # --- 埋設体積を計算 ---
    vol_above, vol_below = calculate_volumes(frustum_verts, get_plane_z)
    pillar_volumes[pillar_id] = {'above': vol_above, 'below': vol_below}

    # --- 敷地で分割して描画 ---
    verts, above_faces, below_faces = clip_mesh_by_plane(frustum_verts, frustum_faces, get_plane_z)
    if above_faces.size > 0:
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=above_faces[:,0],j=above_faces[:,1],k=above_faces[:,2],color='limegreen',opacity=0.3))
    if below_faces.size > 0:
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=below_faces[:,0],j=below_faces[:,1],k=below_faces[:,2],color='sienna',opacity=1.0))

    # --- 土台円柱 ---
    verts, faces = create_cylinder_mesh(base_pos, config['base_cyl_r'], config['base_cyl_h'])
    fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='darkgrey'))
    
    # --- メイン円柱 ---
    verts, faces = create_cylinder_mesh(main_pos, config['main_cyl_r'], config['main_cyl_h'])
    fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='lightslategray'))
    
    # 上部の赤い線
    line_start = [main_pos[0], main_pos[1], main_pos[2] + config['main_cyl_h']]
    line_end = [line_start[0], line_start[1], line_start[2] + 1.5]
    fig.add_trace(go.Scatter3d(x=[line_start[0],line_end[0]],y=[line_start[1],line_end[1]],z=[line_start[2],line_end[2]],mode='lines',line=dict(color='red',width=7)))

# レイアウト設定
fig.update_layout(title_text="敷地と柱の基本表示", scene=dict(xaxis=dict(title='X (m)',range=[-10,10]),yaxis=dict(title='Y (m)',range=[-10,10]),zaxis=dict(title='Z (m)',range=[-10,10]),aspectratio=dict(x=1,y=1,z=1)), margin=dict(l=0,r=0,b=0,t=40), showlegend=False)

st.plotly_chart(fig, use_container_width=True)
st.divider()

# --- 操作パネル (画面下部) ---
st.subheader("各脚の操作と基礎体積")
cols = st.columns(len(pillars_config))
for col, pillar_id in zip(cols, pillars_config.keys()):
    with col:
        st.markdown(f"**{pillar_id}脚**")
        st.markdown("地上部の体積")
        st.subheader(f"{pillar_volumes.get(pillar_id, {}).get('above', 0):.2f} m³")
        st.markdown("埋設部の体積")
        st.subheader(f"{pillar_volumes.get(pillar_id, {}).get('below', 0):.2f} m³")
        up_down_cols = st.columns(2)
        with up_down_cols[0]:
            if st.button(f"⬆️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] += 0.5
                st.rerun()
        with up_down_cols[1]:
            if st.button(f"⬇️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] -= 0.5
                st.rerun()
