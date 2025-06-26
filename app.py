import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay
from streamlit_plotly_events import plotly_events # この行は将来のために残します
import pandas as pd

# --- Streamlitページの基本設定 ---
st.set_page_config(layout="wide", page_title="Site Viewer")

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

# --- データ定義 ---
def get_default_site_data():
    """デフォルトの敷地データを生成する"""
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y)
    z = get_plane_z(xv, yv)
    return np.c_[xv.ravel(), yv.ravel(), z.ravel()]

# --- メインアプリケーション ---
st.title("敷地3Dビューア")
st.info("敷地のみを描画する安定バージョンです。")

# データを生成
site_vertices = get_default_site_data()

# --- 3Dグラフ描画 ---
fig = go.Figure()

# 敷地
if site_vertices is not None and site_vertices.size > 0:
    verts, faces = create_mesh_from_vertices(site_vertices)
    
    # 頂点と面データが正しく生成されたか確認
    if verts.size > 0 and faces.size > 0:
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='burlywood', # 茶色
            opacity=0.8,
            name="Site"
        ))

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

