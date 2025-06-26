import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Site and Pillar Viewer")

# --- セッション状態の初期化 ---
def init_session_state():
    if 'pillar_offsets' not in st.session_state:
        st.session_state.pillar_offsets = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}

# --- 3Dデータを作成・計算する関数 ---
def get_plane_z(x, y, slope_degrees=30):
    """指定された傾斜の平面上のZ座標を計算する"""
    slope_rad = np.deg2rad(slope_degrees)
    return y * np.tan(slope_rad)

def create_mesh_from_vertices(vertices):
    """頂点データからDelaunay三角形分割を用いてメッシュを生成する"""
    try:
        if vertices is None or vertices.shape[0] < 3: return np.array([]), np.array([])
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        return vertices, tri.simplices
    except Exception as e:
        st.error(f"メッシュの生成に失敗しました: {e}"); return np.array([]), np.array([])

def create_cylinder_mesh(center_pos, radius, height, n_segments=32):
    """円柱のメッシュを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x_c, y_c = radius * np.cos(theta), radius * np.sin(theta)
    
    verts = []
    # Bottom circle vertices
    for i in range(n_segments): verts.append([x_c[i], y_c[i], 0])
    # Top circle vertices
    for i in range(n_segments): verts.append([x_c[i], y_c[i], height])
    # Center points for caps
    verts.extend([[0, 0, 0], [0, 0, height]])
    verts = np.array(verts, dtype=float) + np.array(center_pos, dtype=float)
    
    faces = []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        top_i, top_next_i = i + n_segments, next_i + n_segments
        # Side faces
        faces.append([i, next_i, top_next_i])
        faces.append([i, top_next_i, top_i])
        # Bottom cap
        faces.append([next_i, i, 2 * n_segments])
        # Top cap
        faces.append([top_i, top_next_i, 2 * n_segments + 1])
    return verts, np.array(faces, dtype=int)

def create_frustum_mesh(center_pos, bottom_radius, top_radius, height, n_segments=32):
    """円錐台のメッシュを生成する"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    
    verts = []
    # Bottom circle vertices (indices 0 to n-1)
    for angle in theta:
        verts.append([bottom_radius * np.cos(angle), bottom_radius * np.sin(angle), 0])
    # Top circle vertices (indices n to 2n-1)
    for angle in theta:
        verts.append([top_radius * np.cos(angle), top_radius * np.sin(angle), height])
    
    # Add center points for caps
    verts.append([0, 0, 0])      # Bottom center (index 2*n_segments)
    verts.append([0, 0, height]) # Top center (index 2*n_segments + 1)
    
    verts = np.array(verts, dtype=float) + np.array(center_pos)
    
    # Generate faces
    faces = []
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        top_i = i + n_segments
        top_i_next = i_next + n_segments
        
        # Side faces
        faces.append([i, i_next, top_i_next])
        faces.append([i, top_i_next, top_i])
        # Bottom cap
        faces.append([i_next, i, 2 * n_segments])
        # Top cap
        faces.append([top_i, top_i_next, 2 * n_segments + 1])
        
    return verts, np.array(faces, dtype=int)


def get_intersection_polygon(vertices, plane_func):
    """円錐台と平面の交差ポリゴンを計算する"""
    intersection_points = []
    
    # 円錐台の側面を構成する全ての辺で交差判定を行う
    # 頂点の数は (セグメント数 * 2 + 2)
    n_segments = (len(vertices) - 2) // 2
    for i in range(n_segments):
        p1 = vertices[i] # 底面の頂点
        p2 = vertices[i + n_segments] # 上面の頂点
        
        # 各頂点が平面の上にあるか下にあるかを判定
        d1 = p1[2] - plane_func(p1[0], p1[1])
        d2 = p2[2] - plane_func(p2[0], p2[1])
        
        # 辺が平面をまたいでいる場合、交点を計算
        if d1 * d2 < 0:
            ratio = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
            intersect_pt = p1 + ratio * (p2 - p1)
            intersection_points.append(intersect_pt)
    
    if len(intersection_points) < 3: return np.array([])
    
    # 交差点を順序付けしてポリゴンを作成
    points = np.array(intersection_points)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_points = points[np.argsort(angles)]
    
    # 線を閉じるために始点を末尾に追加
    return np.vstack([sorted_points, sorted_points[0]])

def calculate_volumes(verts, plane_func, samples=5000):
    """モンテカルロ法で平面の上下の体積を概算する"""
    if verts is None or verts.size == 0: return 0, 0
    min_c, max_c = verts.min(axis=0), verts.max(axis=0)
    dims = max_c - min_c
    bbox_volume = np.prod(dims)
    if bbox_volume == 0: return 0, 0
    
    # バウンディングボックス内にランダムな点を生成
    random_points = np.random.rand(samples, 3) * dims + min_c
    # 各点における平面のZ座標
    plane_z_at_points = plane_func(random_points[:, 0], random_points[:, 1])
    
    # 点が平面の上にあるか下にあるかを判定
    is_above = random_points[:, 2] >= plane_z_at_points
    is_below = ~is_above
    
    # 点の比率から体積を概算
    vol_above = bbox_volume * (np.sum(is_above) / samples)
    vol_below = bbox_volume * (np.sum(is_below) / samples)
    
    return vol_above, vol_below

# --- データ定義 ---
def get_default_site_data():
    """敷地の頂点データを生成する"""
    x = np.linspace(-10, 10, 20); y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y); z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

def get_default_pillars_config():
    """柱の基本設定を定義する"""
    dist = 7.0 / 2.0; top_r = 1.0; bottom_r = top_r * 3.0
    return {
        'A': {'pos': [-dist, dist], 'foundation_h': 3.0, 'foundation_r_bottom': bottom_r, 'foundation_r_top': top_r, 'base_cyl_r': top_r, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
        'B': {'pos': [dist, dist], 'foundation_h': 3.0, 'foundation_r_bottom': bottom_r, 'foundation_r_top': top_r, 'base_cyl_r': top_r, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
        'C': {'pos': [dist, -dist], 'foundation_h': 3.0, 'foundation_r_bottom': bottom_r, 'foundation_r_top': top_r, 'base_cyl_r': top_r, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
        'D': {'pos': [-dist, -dist], 'foundation_h': 3.0, 'foundation_r_bottom': bottom_r, 'foundation_r_top': top_r, 'base_cyl_r': top_r, 'base_cyl_h': 1.5, 'main_cyl_r': 0.5, 'main_cyl_h': 6.0},
    }

# --- メインアプリケーション ---
init_session_state()
st.title("敷地と柱の3Dビューア")
site_vertices = get_default_site_data()
pillars_config = get_default_pillars_config()
fig = go.Figure()

# 敷地メッシュの描画
if site_vertices is not None and site_vertices.size > 0:
    verts, faces = create_mesh_from_vertices(site_vertices)
    if verts.size > 0:
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='burlywood',opacity=0.7, name='Site'))

pillar_volumes = {}
# 各柱の描画
for pillar_id, config in pillars_config.items():
    x, y = config['pos']; z_off = st.session_state.pillar_offsets.get(pillar_id, 0.0)
    
    # Z座標の計算
    total_h = config['foundation_h'] + config['base_cyl_h'] + config['main_cyl_h']
    init_z = get_plane_z(x, y) - (total_h * 4/5)
    
    foundation_pos_z = init_z + z_off
    # ▼▼▼【変更点】ベース円柱の開始高さを、円錐台の底面（短円）に合わせる ▼▼▼
    base_pos_z = foundation_pos_z
    main_pos_z = base_pos_z + config['base_cyl_h']
    
    foundation_pos = [x, y, foundation_pos_z]; base_pos = [x, y, base_pos_z]; main_pos = [x, y, main_pos_z]
    
    # 円錐台の描画（逆さのまま）
    frustum_verts, frustum_faces = create_frustum_mesh(
        foundation_pos, 
        config['foundation_r_top'],      # 元の上面の半径を底面に
        config['foundation_r_bottom'],   # 元の底面の半径を上面に
        config['foundation_h']
    )
    
    fig.add_trace(go.Mesh3d(x=frustum_verts[:,0], y=frustum_verts[:,1], z=frustum_verts[:,2], i=frustum_faces[:,0],j=frustum_faces[:,1],k=frustum_faces[:,2], color='limegreen', opacity=0.3))
    
    # 交差線の描画
    intersection_line = get_intersection_polygon(frustum_verts, get_plane_z)
    if intersection_line.size > 0:
        fig.add_trace(go.Scatter3d(x=intersection_line[:,0], y=intersection_line[:,1], z=intersection_line[:,2], mode='lines', line=dict(color='orange', width=8)))

    # 体積の計算
    vol_above, vol_below = calculate_volumes(frustum_verts, get_plane_z)
    pillar_volumes[pillar_id] = {'above': vol_above, 'below': vol_below}

    # 他の柱部分の描画
    verts, faces = create_cylinder_mesh(base_pos, config['base_cyl_r'], config['base_cyl_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='darkgrey'))
    verts, faces = create_cylinder_mesh(main_pos, config['main_cyl_r'], config['main_cyl_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='lightslategray'))
    
    # 柱上部の赤い線の描画
    line_start = [main_pos[0], main_pos[1], main_pos[2] + config['main_cyl_h']]; line_end = [line_start[0], line_start[1], line_start[2] + 1.5]
    fig.add_trace(go.Scatter3d(x=[line_start[0],line_end[0]],y=[line_start[1],line_end[1]],z=[line_start[2],line_end[2]],mode='lines',line=dict(color='red',width=7)))

# グラフのレイアウト設定
fig.update_layout(
    title_text="敷地と柱の基本表示（円錐台反転・位置調整）",
    scene=dict(
        xaxis=dict(title='X (m)',range=[-10,10]),
        yaxis=dict(title='Y (m)',range=[-10,10]),
        zaxis=dict(title='Z (m)',range=[-10,10]),
        aspectratio=dict(x=1,y=1,z=1)
    ),
    margin=dict(l=0,r=0,b=0,t=40),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# UI（操作パネルと体積表示）
st.subheader("各脚の操作と基礎体積")
cols = st.columns(len(pillars_config))
for col, pillar_id in zip(cols, pillars_config.keys()):
    with col:
        st.markdown(f"**{pillar_id}脚**")
        st.markdown("地上部の体積"); st.subheader(f"{pillar_volumes.get(pillar_id, {}).get('above', 0):.2f} m³")
        st.markdown("埋設部の体積"); st.subheader(f"{pillar_volumes.get(pillar_id, {}).get('below', 0):.2f} m³")
        up_down_cols = st.columns(2)
        with up_down_cols[0]:
            if st.button(f"⬆️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] += 0.5
                st.rerun()
        with up_down_cols[1]:
            if st.button(f"⬇️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] -= 0.5
                st.rerun()
