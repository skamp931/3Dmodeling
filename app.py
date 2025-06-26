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
    points_2d = vertices[:, :2]
    tri = Delaunay(points_2d)
    return vertices, tri.simplices

def create_cylinder_mesh(center_pos, radius, height, n_segments=32):
    theta = np.linspace(0, 2 * np.pi, n_segments)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    verts = np.array([[x[i], y[i], 0] for i in range(n_segments)] + [[x[i], y[i], height] for i in range(n_segments)], dtype=float) + np.array(center_pos, dtype=float)
    faces = []
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.extend([[i, i_next, i_next + n_segments], [i, i_next + n_segments, i + n_segments]])
    return verts, np.array(faces)

def create_frustum_mesh(center_pos, bottom_radius, top_radius, height, n_segments=32):
    theta = np.linspace(0, 2 * np.pi, n_segments)
    xb, yb = bottom_radius * np.cos(theta), bottom_radius * np.sin(theta)
    xt, yt = top_radius * np.cos(theta), top_radius * np.sin(theta)
    verts = np.array([[xb[i], yb[i], 0] for i in range(n_segments)] + [[xt[i], yt[i], height] for i in range(n_segments)], dtype=float) + np.array(center_pos, dtype=float)
    faces = []
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.extend([[i, i_next, i_next + n_segments], [i, i_next + n_segments, i + n_segments]])
    return verts, np.array(faces)

def calculate_buried_volume_for_one_pillar(buried_components_verts, plane_func, samples=5000):
    if not buried_components_verts: return 0
    all_vertices = np.vstack(buried_components_verts)
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
    xv, yv = np.meshgrid(x, y)
    z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

def get_predefined_pillar_models():
    dist = 7.0 / 2.0
    base = {'pos': [-dist, dist], 'frustum_r_bottom': 2.5}
    models = {
        "Model A (標準)": {
            'A': {**base, 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25},
            'B': {**base, 'pos': [dist, dist], 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25},
            'C': {**base, 'pos': [dist, -dist], 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25},
            'D': {**base, 'pos': [-dist, -dist], 'frustum_h': 2.0, 'base_cyl_h': 1.0, 'main_cyl_h': 5.0, 'frustum_r_top': 0.5, 'base_cyl_r': 0.5, 'main_cyl_r': 0.25},
        },
        "Model B (背高)": {
            'A': {**base, 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2},
            'B': {**base, 'pos': [dist, dist], 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2},
            'C': {**base, 'pos': [dist, -dist], 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2},
            'D': {**base, 'pos': [-dist, -dist], 'frustum_h': 3.0, 'base_cyl_h': 1.5, 'main_cyl_h': 8.0, 'frustum_r_top': 0.4, 'base_cyl_r': 0.4, 'main_cyl_r': 0.2},
        },
        "Model C (寸胴)": {
            'A': {**base, 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0},
            'B': {**base, 'pos': [dist, dist], 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0},
            'C': {**base, 'pos': [dist, -dist], 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0},
            'D': {**base, 'pos': [-dist, -dist], 'frustum_r_bottom': 3.0, 'frustum_h': 1.5, 'base_cyl_h': 1.0, 'main_cyl_h': 4.0, 'frustum_r_top': 1.5, 'base_cyl_r': 1.5, 'main_cyl_r': 1.0},
        }
    }
    return models

def create_pillar_config_from_df(df):
    config = {}
    try:
        df.set_index('id', inplace=True)
        for pillar_id, row in df.iterrows(): config[pillar_id] = row.to_dict()
        return config
    except Exception as e:
        st.error(f"柱CSVファイルの形式が正しくありません: {e}")
        return None

# --- サイドバーUI ---
init_session_state()
st.sidebar.title("🛠️ 設定とツール")
st.sidebar.subheader("1. データソース")
data_source = st.sidebar.radio("表示するデータを選択", ["デフォルト設定", "ファイルから読み込み"], key="data_source_radio", on_change=reset_scene_state)

site_vertices = None
pillars_config = None

if data_source == "ファイルから読み込み":
    st.sidebar.info("CSVファイルをアップロードしてください。")
    uploaded_site_file = st.sidebar.file_uploader("敷地データ (site.csv)", type="csv")
    uploaded_pillars_file = st.sidebar.file_uploader("柱データ (pillars.csv)", type="csv")
    
    site_vertices = get_default_site_data()
    if uploaded_site_file:
        df_site = pd.read_csv(uploaded_site_file)
        if {'x', 'y', 'z'}.issubset(df_site.columns): site_vertices = df_site[['x', 'y', 'z']].values
        else: st.error("敷地ファイルに 'x', 'y', 'z' 列が必要です。")

    if uploaded_pillars_file:
        df_pillars = pd.read_csv(uploaded_pillars_file)
        pillars_config = create_pillar_config_from_df(df_pillars)
    else:
        st.warning("柱データがアップロードされていません。デフォルトモデルを表示します。")
        pillars_config = get_predefined_pillar_models()["Model A (標準)"]

else: # デフォルト設定
    models = get_predefined_pillar_models()
    selected_model = st.sidebar.selectbox("柱のモデルを選択", options=list(models.keys()), key="model_select", index=list(models.keys()).index(st.session_state.selected_model))
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        reset_scene_state()
        st.rerun()

    pillars_config = models[st.session_state.selected_model]
    site_vertices = get_default_site_data()

st.sidebar.subheader("2. 描画ツール")
with st.sidebar.expander("✏️ 画面上で線を描画", expanded=True):
    st.session_state.drawing_mode = st.toggle("描画モードを有効にする", value=st.session_state.drawing_mode)
    if st.button("全ての描画線を削除"): st.session_state.lines, st.session_state.drawing_points = [], []; st.rerun()

# --- メイン画面 ---
st.title("カスタム3Dビルダー")
if pillars_config:
    st.subheader("各脚の操作と埋設体積")
    cols = st.columns(len(pillars_config))
    for i, (pillar_id, config) in enumerate(pillars_config.items()):
        with cols[i]:
            st.markdown(f"**{pillar_id}脚**")
            # === エラー修正箇所 ===
            # st.rerun() を削除し、Streamlitの自然な再描画に任せます
            if st.button(f"⬆️##{pillar_id}",use_container_width=True):
                st.session_state.pillar_offsets[pillar_id]+=0.5
            if st.button(f"⬇️##{pillar_id}",use_container_width=True):
                st.session_state.pillar_offsets[pillar_id]-=0.5
            
            x_pos, y_pos = config['pos']; z_offset = st.session_state.pillar_offsets[pillar_id]
            total_h = config['frustum_h']+config['base_cyl_h']+config['main_cyl_h']
            initial_z_base = get_plane_z(x_pos,y_pos) - (total_h*4/5)
            frustum_pos = [x_pos,y_pos,initial_z_base+z_offset]
            base_cyl_pos = [frustum_pos[0],frustum_pos[1],frustum_pos[2]+config['frustum_h']]
            v1,_=create_frustum_mesh(frustum_pos, config['frustum_r_bottom'], config['frustum_r_top'], config['frustum_h'])
            v2,_=create_cylinder_mesh(base_cyl_pos, config['base_cyl_r'], config['base_cyl_h'])
            vol=calculate_buried_volume_for_one_pillar([v1,v2], get_plane_z)
            # st.metricの引数をキーワード引数で明示的に指定
            st.metric(label="埋設体積", value=f"{vol:.2f} m³", key=f"vol_{pillar_id}")

# --- 3Dグラフ描画 ---
fig = go.Figure()
if site_vertices is not None:
    v,f=create_mesh_from_vertices(site_vertices); fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='burlywood',opacity=0.7))
if pillars_config:
    for pillar_id,config in pillars_config.items():
        x,y=config['pos']; z_off=st.session_state.pillar_offsets[pillar_id]; total_h=config['frustum_h']+config['base_cyl_h']+config['main_cyl_h']
        init_z=get_plane_z(x,y)-(total_h*4/5)
        f_pos=[x,y,init_z+z_off]; b_pos=[f_pos[0],f_pos[1],f_pos[2]+config['frustum_h']]; m_pos=[b_pos[0],b_pos[1],b_pos[2]+config['base_cyl_h']]
        l_s=[m_pos[0],m_pos[1],m_pos[2]+config['main_cyl_h']]; l_e=[l_s[0],l_s[1],l_s[2]+1.5]
        v,f=create_frustum_mesh(f_pos,config['frustum_r_bottom'],config['frustum_r_top'],config['frustum_h']); fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='gray'))
        v,f=create_cylinder_mesh(b_pos,config['base_cyl_r'],config['base_cyl_h']); fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='darkgrey'))
        v,f=create_cylinder_mesh(m_pos,config['main_cyl_r'],config['main_cyl_h']); fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='lightslategray'))
        fig.add_trace(go.Scatter3d(x=[l_s[0],l_e[0]],y=[l_s[1],l_e[1]],z=[l_s[2],l_e[2]],mode='lines',line=dict(color='red',width=7)))
if st.session_state.drawing_points: fig.add_trace(go.Scatter3d(x=[st.session_state.drawing_points[0]['x']],y=[st.session_state.drawing_points[0]['y']],z=[st.session_state.drawing_points[0]['z']],mode='markers',marker=dict(color='magenta',size=10,symbol='cross')))
for line in st.session_state.lines: fig.add_trace(go.Scatter3d(x=[line["start"]['x'],line["end"]['x']],y=[line["start"]['y'],line["end"]['y']],z=[line["start"]['z'],line["end"]['z']],mode='lines',line=dict(color='cyan',width=5)))
fig.update_layout(scene=dict(xaxis=dict(title='X (m)',range=[-10,10]),yaxis=dict(title='Y (m)',range=[-10,10]),zaxis=dict(title='Z (m)',range=[-10,10]),aspectratio=dict(x=1,y=1,z=1)),margin=dict(l=0,r=0,b=0,t=5),showlegend=False)

selected_points=plotly_events(fig,click_event=True,key="plotly_click")
if selected_points and st.session_state.drawing_mode:
    p=selected_points[0]
    if not st.session_state.drawing_points: st.session_state.drawing_points.append(p)
    else: st.session_state.lines.append({"start":st.session_state.drawing_points.pop(0),"end":p})
    st.rerun()
