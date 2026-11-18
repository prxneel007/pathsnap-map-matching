"""PathSnap evaluator UI (laptop 2). Run: streamlit run ui/app.py  -- then set the API URL to http://<laptop-1-ip>:8000"""
import os
import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
try:
    info = requests.get(f"{api}/trips", timeout=10).json()
    st.sidebar.success(f"Connected: {len(info['trips'])} trips in '{info['data_dir']}'")
except Exception as e:  # show the problem instead of crashing the page
    st.sidebar.error(f"Cannot reach API at {api}: {e}")
    st.stop()

trip = st.selectbox("Evaluation trip", info["trips"])
if st.button("Run map-matching", type="primary"):
    t0 = time.time()
    r = requests.get(f"{api}/predict/{trip}", timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}"); st.stop()
    st.session_state["res"] = r.json(); st.session_state["rtt"] = time.time() - t0

res = st.session_state.get("res")
if res and res["trip_id"] == trip:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"
    h, c = res["hmm"], res["clf"]
    cols = st.columns(6)
    cols[0].metric("Point accuracy (HMM)", pct(h.get("point_accuracy")),
                   None if h.get("point_accuracy") is None else f"{100 * (h['point_accuracy'] - c['point_accuracy']):+.2f} pts vs classifier only")
    cols[1].metric("Road-level accuracy (HMM)", pct(h.get("road_accuracy")), help="Correct road in either direction")
    cols[2].metric("Classifier-only accuracy", pct(c.get("point_accuracy")))
    cols[3].metric("Route arc F1 (HMM)", pct(h.get("route_arc_f1")))
    cols[4].metric("Route continuity (HMM)", pct(h.get("route_continuity")))
    cols[5].metric("GPS points", f"{res['gps_points']:,}")
    st.caption(f"API prediction time {res['prediction_seconds']} s · round trip {st.session_state['rtt']:.1f} s · "
               f"{res['segments']} segment(s) · {res['chain_breaks']} chain break(s) · accuracy computed server-side")

    m = folium.Map(tiles=None)
    folium.TileLayer(ESRI, attr="Tiles &copy; Esri", name="Esri street map").add_to(m)
    fg = folium.FeatureGroup("GPS observations")
    for lat, lon in res["gps"]:
        folium.CircleMarker([lat, lon], radius=2, color="red", fill=True, weight=0, fill_opacity=.5).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup("Predicted route (HMM)")   # added last so it is drawn above the GPS dots
    folium.PolyLine(res["predicted_route"], color="blue", weight=4, opacity=.9).add_to(fg); fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    lats, lons = [g[0] for g in res["gps"]], [g[1] for g in res["gps"]]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st.markdown("**Map:** <span style='color:red'>●</span> raw GPS &nbsp; <span style='color:blue'>━</span> predicted route", unsafe_allow_html=True)
    st_folium(m, height=600, use_container_width=True, returned_objects=[])
    with st.expander("Raw API response (metrics)"):
        st.json({k: v for k, v in res.items() if k not in ("gps", "predicted_route")})
