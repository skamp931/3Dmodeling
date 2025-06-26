import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay
from streamlit_plotly_events import plotly_events
import pandas as pd

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Custom 3D Builder")

# --- セッション状態の初期化 ---
def init_session_state():
    if 'lines' not in st.session_state: st.session_state.lines = []
    if 'drawing_points' not in st.session_state: st.session_state.drawing_points = []
    if 'drawing_mode' not in st.session_state: st.session_state.drawing_mode = False
    if 'pillar_offsets' not in st.session_state:
        st.session_state.pillar_offsets = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "デフォルト設定"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Model A (標準)"

def reset_scene_state():
    """柱の高さや描画線をリセットする"""
    st.session_state.pillar_offsets = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    st.session_state.lines = []
    st.session_state.drawing_points = []

# --- 3Dデータを作成・計算する関数 ---
def get_plane_z(x, y, slope_degrees=30):
    slope_rad = np.deg2rad(slope_degrees)
    return y * np.tan(slope_rad)

def create_mesh_from_vertices(vertices):
    """頂点群からDelaunay三角分割でメッシュを作成する"""
    try:
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        return vertices, tri.simplices
    except Exception:
        return np.array([]), np.array([])

def create_cylinder_mesh(center_pos, radius, height, n_segments=32):
    """【最終修正版】円柱のメッシュを作成"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    
    verts = []
    # Bottom circle (indices 0 to n-1)
    for i in range(n_segments): verts.append([x[i], y[i], 0])
    # Top circle (indices n to 2n-1)
    for i in range(n_segments): verts.append([x[i], y[i], height])
    verts.append([0, 0, 0])      # Bottom center (index: 2n)
    verts.append([0, 0, height]) # Top center (index: 2n+1)
    
    verts = np.array(verts, dtype=float) + np.array(center_pos, dtype=float)
    
    faces = []
    # Side faces
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([i, next_i, i + n_segments])
        faces.append([next_i, next_i + n_segments, i + n_segments])
    # Bottom cap
    for i in range(n_segments): faces.append([i, (i + 1) % n_segments, 2 * n_segments])
    # Top cap
    for i in range(n_segments): faces.append([i + n_segments, ((i + 1) % n_segments) + n_segments, 2 * n_segments + 1])
        
    return verts, np.array(faces, dtype=int)

def create_frustum_mesh(center_pos, bottom_radius, top_radius, height, n_segments=32):
    """【最終修正版】円錐台のメッシュを作成"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    xb, yb = bottom_radius * np.cos(theta), bottom_radius * np.sin(theta)
    xt, yt = top_radius * np.cos(theta), top_radius * np.sin(theta)
    
    verts = []
    # Bottom circle (indices 0 to n-1)
    for i in range(n_segments): verts.append([xb[i], yb[i], 0])
    # Top circle (indices n to 2n-1)
    for i in range(n_segments): verts.append([xt[i], yt[i], height])
    verts.append([0, 0, 0])      # Bottom center (index: 2n)
    verts.append([0, 0, height]) # Top center (index: 2n+1)
    
    verts = np.array(verts, dtype=float) + np.array(center_pos, dtype=float)

    faces = []
    # Side faces
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([i, next_i, i + n_segments])
        faces.append([next_i, next_i + n_segments, i + n_segments])
    # Bottom cap
    for i in range(n_segments): faces.append([i, (i + 1) % n_segments, 2 * n_segments])
    # Top cap
    for i in range(n_segments): faces.append([i + n_segments, ((i + 1) % n_segments) + n_segments, 2 * n_segments + 1])
        
    return verts, np.array(faces, dtype=int)

def calculate_buried_volume_for_one_pillar(verts_list, plane_func, samples=5000):
    if not verts_list: return 0
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
    x = np.linspace(-10, 10, 20); y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y); z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

def get_predefined_pillar_models():
    dist = 7.0 / 2.0; base = {'pos': [-dist, dist], 'frustum_r_bottom': 2.5}
    models = { "Model A (標準)": { 'A': {**base, 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25}, 'B': {**base, 'pos': [dist, dist], 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25}, 'C': {**base, 'pos': [dist, -dist], 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25}, 'D': {**base, 'pos': [-dist, -dist], 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25}}, "Model B (背高)": { 'A': {**base, 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2}, 'B': {**base, 'pos': [dist, dist], 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2}, 'C': {**base, 'pos': [dist, -dist], 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2}, 'D': {**base, 'pos': [-dist, -dist], 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2}}, "Model C (寸胴)": { 'A': {**base, 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0}, 'B': {**base, 'pos': [dist, dist], 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0}, 'C': {**base, 'pos': [dist, -dist], 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0}, 'D': {**base, 'pos': [-dist, -dist], 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0}}, }
    return models

def create_pillar_config_from_df(df):
    config = {}; required_cols = {'id', 'x', 'y', 'frustum_h', 'base_cyl_h', 'main_cyl_h', 'frustum_r_bottom', 'frustum_r_top', 'base_cyl_r', 'main_cyl_r'}
    try:
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns); st.error(f"柱CSVファイルに必須の列がありません: {', '.join(missing)}"); return None
        df.set_index('id', inplace=True)
        for pillar_id, row in df.iterrows(): config[str(pillar_id)] = {'pos': [row['x'], row['y']], **row.to_dict()}
        return config
    except Exception as e: st.error(f"柱CSVファイルの処理中にエラーが発生しました: {e}"); return None

# --- UIと描画 ---
init_session_state()
st.sidebar.title("🛠️ 設定とツール"); st.sidebar.subheader("1. データソース")
data_source = st.sidebar.radio("表示するデータを選択", ["デフォルト設定", "ファイルから読み込み"], key="data_source_radio", on_change=reset_scene_state)
site_vertices, pillars_config = None, None
if data_source == "ファイルから読み込み":
    st.sidebar.info("CSVファイルをアップロードしてください。")
    uploaded_site_file = st.sidebar.file_uploader("敷地データ (site.csv)", type="csv")
    uploaded_pillars_file = st.sidebar.file_uploader("柱データ (pillars.csv)", type="csv")
    site_vertices = get_default_site_data()
    if uploaded_site_file:
        try:
            df_site = pd.read_csv(uploaded_site_file)
            if {'x', 'y', 'z'}.issubset(df_site.columns): site_vertices = df_site[['x', 'y', 'z']].values
            else: st.error("敷地ファイルに 'x', 'y', 'z' 列が必要です。")
        except Exception as e: st.error(f"敷地ファイルの読み込みエラー: {e}")
    if uploaded_pillars_file:
        try: df_pillars = pd.read_csv(uploaded_pillars_file); pillars_config = create_pillar_config_from_df(df_pillars)
        except Exception as e: st.error(f"柱ファイルの読み込みエラー: {e}")
    if pillars_config is None: st.warning("柱データが読み込めないため、デフォルトモデルを表示します。"); pillars_config = get_predefined_pillar_models()["Model A (標準)"]
else: 
    models = get_predefined_pillar_models()
    selected_model = st.sidebar.selectbox("柱のモデルを選択", options=list(models.keys()), key="model_select", index=list(models.keys()).index(st.session_state.selected_model))
    if selected_model != st.session_state.selected_model: st.session_state.selected_model = selected_model; reset_scene_state(); st.rerun()
    pillars_config = models[st.session_state.selected_model]
    site_vertices = get_default_site_data()

st.sidebar.subheader("2. 描画ツール")
with st.sidebar.expander("✏️ 画面上で線を描画", expanded=True):
    st.session_state.drawing_mode = st.toggle("描画モードを有効にする", value=st.session_state.drawing_mode)
    if st.button("全ての描画線を削除"): st.session_state.lines, st.session_state.drawing_points = [], []; st.rerun()

st.title("カスタム3Dビルダー")
if pillars_config:
    st.subheader("各脚の操作と埋設体積")
    cols = st.columns(len(pillars_config))
    for i, (pillar_id, config) in enumerate(cols):
        with config:
            st.markdown(f"**{pillar_id}脚**")
            if st.button(f"⬆️##{pillar_id}",use_container_width=True): st.session_state.pillar_offsets[pillar_id]+=0.5; st.rerun()
            if st.button(f"⬇️##{pillar_id}",use_container_width=True): st.session_state.pillar_offsets[pillar_id]-=0.5; st.rerun()
            x,y=pillars_config[pillar_id]['pos']; z_off=st.session_state.pillar_offsets[pillar_id]
            cfg = pillars_config[pillar_id]
            total_h=cfg['frustum_h']+cfg['base_cyl_h']+cfg['main_cyl_h']
            init_z=get_plane_z(x,y)-(total_h*4/5)
            f_pos=[x,y,init_z+z_off]; b_pos=[f_pos[0],f_pos[1],f_pos[2]+cfg['frustum_h']]
            v1,_=create_frustum_mesh(f_pos,cfg['frustum_r_bottom'],cfg['frustum_r_top'],cfg['frustum_h'])
            v2,_=create_cylinder_mesh(b_pos,cfg['base_cyl_r'],cfg['base_cyl_h'])
            vol=calculate_buried_volume_for_one_pillar([v1,v2], get_plane_z)
            st.markdown("埋設体積"); st.subheader(f"{vol:.2f} m³")

# 3Dグラフ描画
fig = go.Figure()
if site_vertices is not None and len(site_vertices) > 0:
    v, f = create_mesh_from_vertices(site_vertices)
    if v.size > 0: fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='burlywood',opacity=0.7))

if pillars_config:
    for pillar_id, config in pillars_config.items():
        x,y=config['pos']; z_off=st.session_state.pillar_offsets[pillar_id]; total_h=config['frustum_h']+config['base_cyl_h']+config['main_cyl_h']
        init_z=get_plane_z(x,y)-(total_h*4/5)
        f_pos=[x,y,init_z+z_off]; b_pos=[f_pos[0],f_pos[1],f_pos[2]+config['frustum_h']]; m_pos=[b_pos[0],b_pos[1],b_pos[2]+config['base_cyl_h']]
        l_s=[m_pos[0],m_pos[1],m_pos[2]+config['main_cyl_h']]; l_e=[l_s[0],l_s[1],l_s[2]+1.5]
        
        verts, faces = create_frustum_mesh(f_pos,config['frustum_r_bottom'],config['frustum_r_top'],config['frustum_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='gray'))
        verts, faces = create_cylinder_mesh(b_pos,config['base_cyl_r'],config['base_cyl_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='darkgrey'))
        verts, faces = create_cylinder_mesh(m_pos,config['main_cyl_r'],config['main_cyl_h']); fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='lightslategray'))
        fig.add_trace(go.Scatter3d(x=[l_s[0],l_e[0]],y=[l_s[1],l_e[1]],z=[l_s[2],l_e[2]],mode='lines',line=dict(color='red',width=7)))

if st.session_state.drawing_points: fig.add_trace(go.Scatter3d(x=[st.session_state.drawing_points[0]['x']],y=[st.session_state.drawing_points[0]['y']],z=[st.session_state.drawing_points[0]['z']],mode='markers',marker=dict(color='magenta',size=10,symbol='cross')))
for line in st.session_state.lines: fig.add_trace(go.Scatter3d(x=[line["start"]['x'],line["end"]['x']],y=[line["start"]['y'],line["end"]['y']],z=[line["start"]['z'],line["end"]['z']],mode='lines',line=dict(color='cyan',width=5)))
fig.update_layout(scene=dict(xaxis=dict(title='X(m)',range=[-10,10]),yaxis=dict(title='Y(m)',range=[-10,10]),zaxis=dict(title='Z(m)',range=[-10,10]),aspectratio=dict(x=1,y=1,z=1)),margin=dict(l=0,r=0,b=0,t=40),showlegend=False)

selected_points=plotly_events(fig,click_event=True,key="plotly_click")
if selected_points and st.session_state.drawing_mode:
    p=selected_points[0]
    if not st.session_state.drawing_points:st.session_state.drawing_points.append(p)
    else:st.session_state.lines.append({"start":st.session_state.drawing_points.pop(0),"end":p})
    st.rerun()
