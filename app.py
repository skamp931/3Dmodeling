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
        if vertices is None or vertices.shape[0] < 3:
            return np.array([]), np.array([])
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        return vertices, tri.simplices
    except Exception as e:
        st.error(f"メッシュの生成に失敗しました: {e}")
        return np.array([]), np.array([])

def create_elliptical_cylinder_mesh(center_pos, radius_x, radius_y, height, n_segments=32):
    """楕円円柱の頂点と面データを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x_c = radius_x * np.cos(theta)
    y_c = radius_y * np.sin(theta)
    
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

def create_slanted_elliptical_frustum_mesh(center_pos, bottom_radius_x, bottom_radius_y, top_radius_x, top_radius_y, top_z, plane_func, n_segments=32):
    """敷地面に沿った楕円円錐台のメッシュを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    
    verts = []
    # Top ellipse vertices (at a fixed height)
    xt, yt = top_radius_x * np.cos(theta), top_radius_y * np.sin(theta)
    for i in range(n_segments):
        verts.append([center_pos[0] + xt[i], center_pos[1] + yt[i], top_z])
        
    # Bottom ellipse vertices (Z value follows the plane)
    xb, yb = bottom_radius_x * np.cos(theta), bottom_radius_y * np.sin(theta)
    for i in range(n_segments):
        px, py = center_pos[0] + xb[i], center_pos[1] + yb[i]
        pz = plane_func(px, py)
        verts.append([px, py, pz])
    
    verts = np.array(verts, dtype=float)
    
    faces = []
    # Side faces
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Top circle indices are i, next_i
        # Bottom circle indices are i + n_segments, next_i + n_segments
        faces.extend([
            [i, next_i, i + n_segments],
            [next_i, next_i + n_segments, i + n_segments]
        ])
    
    # Top cap (flat)
    top_center_idx = len(verts)
    verts = np.vstack([verts, [center_pos[0], center_pos[1], top_z]])
    for i in range(n_segments):
        faces.append([i, (i + 1) % n_segments, top_center_idx])

    return verts, np.array(faces, dtype=int)


def calculate_buried_volume(verts_list, plane_func, samples=5000):
    if not verts_list or len(verts_list[0]) == 0: return 0
    all_vertices = np.vstack(verts_list)
    min_c, max_c = all_vertices.min(axis=0), all_vertices.max(axis=0)
    dims, bbox_volume = max_c - min_c, np.prod(max_c - min_c)
    if bbox_volume == 0: return 0
    return bbox_volume

# --- データ定義 ---
def get_default_site_data():
    x = np.linspace(-10, 10, 20); y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y); z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

def get_default_pillars_config():
    dist = 7.0 / 2.0
    rx_ratio = 1.0
    ry_ratio = 1.5
    config = {
        'A': {'pos': [-dist, dist], 'foundation_r_bottom_x': 2.5*rx_ratio, 'foundation_r_bottom_y': 2.5*ry_ratio, 'foundation_r_top_x': 2.0*rx_ratio, 'foundation_r_top_y': 2.0*ry_ratio, 'base_cyl_r_x': 2.0*rx_ratio, 'base_cyl_r_y': 2.0*ry_ratio, 'base_cyl_h': 1.5, 'main_cyl_r_x': 0.5*rx_ratio, 'main_cyl_r_y': 0.5*ry_ratio, 'main_cyl_h': 6.0},
        'B': {'pos': [dist, dist], 'foundation_r_bottom_x': 2.5*rx_ratio, 'foundation_r_bottom_y': 2.5*ry_ratio, 'foundation_r_top_x': 2.0*rx_ratio, 'foundation_r_top_y': 2.0*ry_ratio, 'base_cyl_r_x': 2.0*rx_ratio, 'base_cyl_r_y': 2.0*ry_ratio, 'base_cyl_h': 1.5, 'main_cyl_r_x': 0.5*rx_ratio, 'main_cyl_r_y': 0.5*ry_ratio, 'main_cyl_h': 6.0},
        'C': {'pos': [dist, -dist], 'foundation_r_bottom_x': 2.5*rx_ratio, 'foundation_r_bottom_y': 2.5*ry_ratio, 'foundation_r_top_x': 2.0*rx_ratio, 'foundation_r_top_y': 2.0*ry_ratio, 'base_cyl_r_x': 2.0*rx_ratio, 'base_cyl_r_y': 2.0*ry_ratio, 'base_cyl_h': 1.5, 'main_cyl_r_x': 0.5*rx_ratio, 'main_cyl_r_y': 0.5*ry_ratio, 'main_cyl_h': 6.0},
        'D': {'pos': [-dist, -dist], 'foundation_r_bottom_x': 2.5*rx_ratio, 'foundation_r_bottom_y': 2.5*ry_ratio, 'foundation_r_top_x': 2.0*rx_ratio, 'foundation_r_top_y': 2.0*ry_ratio, 'base_cyl_r_x': 2.0*rx_ratio, 'base_cyl_r_y': 2.0*ry_ratio, 'base_cyl_h': 1.5, 'main_cyl_r_x': 0.5*rx_ratio, 'main_cyl_r_y': 0.5*ry_ratio, 'main_cyl_h': 6.0},
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
    x, y = config['pos']
    z_off = st.session_state.pillar_offsets.get(pillar_id, 0.0)
    total_h = config['base_cyl_h'] + config['main_cyl_h'] # foundation height is not included here for positioning
    init_z = get_plane_z(x, y) - (total_h * 4/5)
    
    # パーツの位置を計算
    base_pos_z = init_z + z_off
    main_pos_z = base_pos_z + config['base_cyl_h']
    
    base_pos = [x, y, base_pos_z]
    main_pos = [x, y, main_pos_z]
    
    # --- 新しい基礎柱（計算対象）---
    foundation_verts, foundation_faces = create_slanted_elliptical_frustum_mesh(
        [x, y, 0], 
        config['foundation_r_bottom_x'], config['foundation_r_bottom_y'],
        config['foundation_r_top_x'], config['foundation_r_top_y'],
        base_pos_z, 
        get_plane_z
    )
    fig.add_trace(go.Mesh3d(x=foundation_verts[:,0], y=foundation_verts[:,1], z=foundation_verts[:,2], i=foundation_faces[:,0],j=foundation_faces[:,1],k=foundation_faces[:,2], color='limegreen', opacity=0.3, name=f"Foundation {pillar_id}"))
    
    # --- 埋設体積を計算 ---
    pillar_volumes[pillar_id] = calculate_buried_volume([foundation_verts], get_plane_z)

    # --- 土台円柱 ---
    verts, faces = create_elliptical_cylinder_mesh(base_pos, config['base_cyl_r_x'], config['base_cyl_r_y'], config['base_cyl_h'])
    fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='darkgrey'))
    
    # --- メイン円柱 ---
    verts, faces = create_elliptical_cylinder_mesh(main_pos, config['main_cyl_r_x'], config['main_cyl_r_y'], config['main_cyl_h'])
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
        st.markdown("基礎部分の体積")
        st.subheader(f"{pillar_volumes.get(pillar_id, 0):.2f} m³")
        up_down_cols = st.columns(2)
        with up_down_cols[0]:
            if st.button(f"⬆️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] += 0.5
                st.rerun()
        with up_down_cols[1]:
            if st.button(f"⬇️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] -= 0.5
                st.rerun()
