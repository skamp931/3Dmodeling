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

def calculate_buried_volume(verts_list, plane_func, samples=5000):
    """与えられた頂点群の埋設体積を計算する"""
    if not verts_list or len(verts_list[0]) == 0: return 0
    all_vertices = np.vstack(verts_list)
    min_c, max_c = all_vertices.min(axis=0), all_vertices.max(axis=0)
    dims, bbox_volume = max_c - min_c, np.prod(max_c - min_c)
    if bbox_volume == 0: return 0
    
    random_points = np.random.rand(samples, 3) * dims + min_c
    plane_z_at_points = plane_func(random_points[:, 0], random_points[:, 1])
    is_below_plane = random_points[:, 2] < plane_z_at_points
    
    return bbox_volume * (np.sum(is_below_plane) / samples)

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

# --- 3Dグラフ描画 ---
fig = go.Figure()

# 敷地
if site_vertices is not None and site_vertices.size > 0:
    verts, faces = create_mesh_from_vertices(site_vertices)
    if verts.size > 0 and faces.size > 0:
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='burlywood', opacity=0.7, name="Site"
        ))

# 柱の描画と体積計算用のデータ準備
pillar_volumes = {}
for pillar_id, config in pillars_config.items():
    x, y = config['pos']
    z_off = st.session_state.pillar_offsets.get(pillar_id, 0.0)
    total_h = config['base_cyl_h'] + config['main_cyl_h']
    init_z = get_plane_z(x, y) - (total_h * 4/5)
    
    # パーツの位置を計算
    base_pos = [x, y, init_z + z_off]
    main_pos = [base_pos[0], base_pos[1], base_pos[2] + config['base_cyl_h']]
    
    # --- ベース円柱（計算対象＆色分け） ---
    base_verts, base_faces = create_cylinder_mesh(base_pos, config['base_cyl_r'], config['base_cyl_h'])
    vertex_colors_base = []
    for v in base_verts:
        site_z = get_plane_z(v[0], v[1])
        if v[2] < site_z:
            vertex_colors_base.append('sienna')  # 埋設部分の色
        else:
            vertex_colors_base.append('limegreen') # 計算対象の地上部分の色
    fig.add_trace(go.Mesh3d(x=base_verts[:,0],y=base_verts[:,1],z=base_verts[:,2],i=base_faces[:,0],j=base_faces[:,1],k=base_faces[:,2],vertexcolor=vertex_colors_base, name=f"Base {pillar_id}"))
    
    # --- 埋設体積を計算（ベース部分のみ対象） ---
    pillar_volumes[pillar_id] = calculate_buried_volume([base_verts], get_plane_z)

    # --- メイン円柱（計算対象外、常に一定の色） ---
    main_verts, main_faces = create_cylinder_mesh(main_pos, config['main_cyl_r'], config['main_cyl_h'])
    fig.add_trace(go.Mesh3d(x=main_verts[:,0],y=main_verts[:,1],z=main_verts[:,2],i=main_faces[:,0],j=main_faces[:,1],k=main_faces[:,2],color='lightslategray', name=f"Main {pillar_id}"))
    
    # 上部の赤い線を描画
    line_start = [main_pos[0], main_pos[1], main_pos[2] + config['main_cyl_h']]
    line_end = [line_start[0], line_start[1], line_start[2] + 1.5]
    fig.add_trace(go.Scatter3d(x=[line_start[0],line_end[0]],y=[line_start[1],line_end[1]],z=[line_start[2],line_end[2]],mode='lines',line=dict(color='red',width=7)))


# レイアウト設定
fig.update_layout(
    title_text="敷地と柱の基本表示",
    scene=dict(
        xaxis=dict(title='X (m)', range=[-10, 10]),
        yaxis=dict(title='Y (m)', range=[-10, 10]),
        zaxis=dict(title='Z (m)', range=[-10, 10]),
        aspectratio=dict(x=1, y=1, z=1)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False
)

# Streamlitで3Dビューアを表示
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 操作パネル (画面下部) ---
st.subheader("各脚の操作と埋設体積")
cols = st.columns(len(pillars_config))
for col, pillar_id in zip(cols, pillars_config.keys()):
    with col:
        st.markdown(f"**{pillar_id}脚**")
        
        # 体積の表示
        st.markdown("埋設体積（土台部分）")
        st.subheader(f"{pillar_volumes.get(pillar_id, 0):.2f} m³")

        # 上下ボタン
        up_down_cols = st.columns(2)
        with up_down_cols[0]:
            if st.button(f"⬆️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] += 0.5
                st.rerun()
        with up_down_cols[1]:
            if st.button(f"⬇️##{pillar_id}", use_container_width=True):
                st.session_state.pillar_offsets[pillar_id] -= 0.5
                st.rerun()
