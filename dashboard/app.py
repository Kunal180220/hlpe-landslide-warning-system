"""
HLPE — Hybrid Landslide Prediction Engine
Main Streamlit Dashboard with State → District → Taluk navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import json
import io
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.admin_hierarchy import (
    get_states, get_districts, get_taluks,
    get_bbox, get_center, get_admin_info,
    ADMIN_HIERARCHY,
)
from engine.slope_units import load_or_generate
from engine.rainfall_fetcher import get_rainfall_for_slope_units
from engine.factor_of_safety import compute_all_scores


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HLPE — Landslide Early Warning",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main theme */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 5px solid #e94560;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #a0aec0; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

    /* Breadcrumb */
    .breadcrumb {
        background: #f0f4f8;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #4a5568;
        margin-bottom: 0.8rem;
        border-left: 3px solid #667eea;
    }

    /* KPI cards */
    .kpi-grid { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        flex: 1;
        min-width: 140px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 3px solid;
        text-align: center;
    }
    .kpi-card.extreme { border-color: #ff2929; }
    .kpi-card.high { border-color: #ff6b6b; }
    .kpi-card.elevated { border-color: #ffa500; }
    .kpi-card.moderate { border-color: #ffd700; }
    .kpi-card.low { border-color: #27ae60; }
    .kpi-card .val { font-size: 2rem; font-weight: 700; }
    .kpi-card .lbl { font-size: 0.75rem; color: #718096; text-transform: uppercase; }

    /* Alert banner */
    .alert-extreme { background: #fff5f5; border: 1px solid #fc8181; border-radius: 8px; padding: 0.7rem 1rem; color: #c53030; }
    .alert-high { background: #fff5f5; border: 1px solid #feb2b2; border-radius: 8px; padding: 0.7rem 1rem; color: #e53e3e; }
    .alert-low { background: #f0fff4; border: 1px solid #9ae6b4; border-radius: 8px; padding: 0.7rem 1rem; color: #276749; }

    /* Slope card */
    .slope-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid;
        cursor: pointer;
    }

    /* Admin level badges */
    .badge-state { background: #667eea; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-district { background: #48bb78; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-taluk { background: #ed8936; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "selected_state": "Kerala",
        "selected_district": None,
        "selected_taluk": None,
        "slope_df": None,
        "computed": False,
        "manual_rainfall": None,
        "uploaded_df": None,
        "use_api": False,  # off by default for speed
        "last_computed": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def alert_color(level: str) -> str:
    return {
        "EXTREME": "#ff2929", "HIGH": "#ff6b6b",
        "ELEVATED": "#ffa500", "MODERATE": "#ffd700", "LOW": "#27ae60"
    }.get(level, "#aaa")


def alert_emoji(level: str) -> str:
    return {"EXTREME": "⚫", "HIGH": "🔴", "ELEVATED": "🟠", "MODERATE": "🟡", "LOW": "🟢"}.get(level, "⚪")


def get_zoom_for_level() -> int:
    if st.session_state.selected_taluk:
        return 12
    elif st.session_state.selected_district:
        return 10
    return 7


def get_current_bbox():
    return get_bbox(
        st.session_state.selected_state,
        st.session_state.selected_district,
        st.session_state.selected_taluk,
    )


def get_current_center():
    return get_center(
        st.session_state.selected_state,
        st.session_state.selected_district,
        st.session_state.selected_taluk,
    )


def build_breadcrumb():
    parts = [f"🗺️ {st.session_state.selected_state}"]
    if st.session_state.selected_district:
        parts.append(st.session_state.selected_district)
    if st.session_state.selected_taluk:
        parts.append(st.session_state.selected_taluk)
    return " → ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — ADMIN NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
        <span style='font-size:2rem'>🏔️</span>
        <h3 style='margin:0.2rem 0 0 0; color:#2d3748;'>HLPE</h3>
        <small style='color:#718096;'>Landslide Early Warning</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 Administrative Level")
    
    # STATE selector
    states = get_states()
    sel_state = st.selectbox(
        "State",
        options=states,
        index=states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0,
        key="state_select",
    )
    if sel_state != st.session_state.selected_state:
        st.session_state.selected_state = sel_state
        st.session_state.selected_district = None
        st.session_state.selected_taluk = None
        st.session_state.computed = False
        st.session_state.slope_df = None
        st.rerun()

    # DISTRICT selector
    districts = ["— All Districts —"] + get_districts(sel_state)
    dist_idx = 0
    if st.session_state.selected_district and st.session_state.selected_district in districts:
        dist_idx = districts.index(st.session_state.selected_district)
    sel_district_raw = st.selectbox("District", options=districts, index=dist_idx, key="dist_select")
    sel_district = None if sel_district_raw.startswith("—") else sel_district_raw
    if sel_district != st.session_state.selected_district:
        st.session_state.selected_district = sel_district
        st.session_state.selected_taluk = None
        st.session_state.computed = False
        st.session_state.slope_df = None
        st.rerun()

    # TALUK selector
    if st.session_state.selected_district:
        taluks = ["— All Taluks —"] + get_taluks(sel_state, st.session_state.selected_district)
        tal_idx = 0
        if st.session_state.selected_taluk and st.session_state.selected_taluk in taluks:
            tal_idx = taluks.index(st.session_state.selected_taluk)
        sel_taluk_raw = st.selectbox("Taluk", options=taluks, index=tal_idx, key="taluk_select")
        sel_taluk = None if sel_taluk_raw.startswith("—") else sel_taluk_raw
        if sel_taluk != st.session_state.selected_taluk:
            st.session_state.selected_taluk = sel_taluk
            st.session_state.computed = False
            st.session_state.slope_df = None
            st.rerun()
    else:
        st.session_state.selected_taluk = None

    st.markdown("---")
    
    # Level indicator
    level = "taluk" if st.session_state.selected_taluk else "district" if st.session_state.selected_district else "state"
    level_colors = {"state": "#667eea", "district": "#48bb78", "taluk": "#ed8936"}
    st.markdown(f"""
    <div style='background:{level_colors[level]}15; border:1px solid {level_colors[level]}; border-radius:8px; padding:0.5rem 0.8rem;'>
        <b style='color:{level_colors[level]};'>📊 Level: {level.title()}</b><br>
        <small style='color:#4a5568;'>
        {st.session_state.selected_state}<br>
        {"📍 " + st.session_state.selected_district if st.session_state.selected_district else "All districts"}<br>
        {"🔍 " + st.session_state.selected_taluk if st.session_state.selected_taluk else "All taluks"}
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    use_api = st.toggle("Live Rainfall API", value=st.session_state.use_api, help="Fetch real rainfall from Open-Meteo")
    st.session_state.use_api = use_api
    
    n_units = st.slider("Slope Units to Generate", 20, 150, 60)
    
    st.markdown("---")
    st.markdown("### 🌧️ Manual Rainfall Override")
    manual_today = st.number_input("Today's Rain (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    if manual_today > 0:
        st.session_state.manual_rainfall = {"today_mm": manual_today}
    
    st.markdown("---")
    
    # Compute button
    if st.button("🔄 Compute Risk Scores", type="primary", use_container_width=True):
        with st.spinner("Loading slope units and computing risks..."):
            df = load_or_generate(
                state=st.session_state.selected_state,
                district=st.session_state.selected_district,
                taluk=st.session_state.selected_taluk,
                uploaded_df=st.session_state.uploaded_df,
                n_units=n_units,
            )
            df = get_rainfall_for_slope_units(
                df,
                manual_rainfall=st.session_state.manual_rainfall,
                use_api=st.session_state.use_api,
            )
            df = compute_all_scores(df)
            st.session_state.slope_df = df
            st.session_state.computed = True
            st.session_state.last_computed = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.success("✅ Risk computed!")
    
    if st.session_state.last_computed:
        st.caption(f"Last updated: {st.session_state.last_computed}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🏔️ HLPE — Hybrid Landslide Prediction Engine</h1>
    <p>AI + Physics hybrid | Slope-specific micro-alerts | Dynamic rainfall | Real-time risk scoring</p>
</div>
""", unsafe_allow_html=True)

# Breadcrumb
st.markdown(f'<div class="breadcrumb">{build_breadcrumb()}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs(["🗺️ Risk Map", "📊 Slope Dashboard", "🌧️ Rainfall", "📁 Upload Data", "📄 Report"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: RISK MAP
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    if not st.session_state.computed or st.session_state.slope_df is None:
        # Show placeholder map centred on selected area
        center = get_current_center()
        zoom = get_zoom_for_level()
        
        m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, tiles="CartoDB positron")
        
        # Draw boundary box
        bbox = get_current_bbox()
        folium.Rectangle(
            bounds=[[bbox[2], bbox[0]], [bbox[3], bbox[1]]],
            color="#667eea", weight=2, fill=True, fill_opacity=0.05,
            tooltip=f"{st.session_state.selected_state} | {st.session_state.selected_district or 'All Districts'}",
        ).add_to(m)
        
        # Add district boundaries if at state level
        if not st.session_state.selected_district:
            for dname, ddata in ADMIN_HIERARCHY.get(st.session_state.selected_state, {}).get("districts", {}).items():
                db = ddata["bbox"]
                folium.Rectangle(
                    bounds=[[db[2], db[0]], [db[3], db[1]]],
                    color="#48bb78", weight=1.5, fill=True, fill_opacity=0.03,
                    tooltip=f"District: {dname}",
                    popup=f"<b>{dname}</b><br>Click sidebar to zoom in",
                ).add_to(m)
        
        # Add taluk boundaries if at district level
        if st.session_state.selected_district and not st.session_state.selected_taluk:
            from engine.admin_hierarchy import get_all_taluks_in_district
            taluks_data = get_all_taluks_in_district(st.session_state.selected_state, st.session_state.selected_district)
            for tname, tdata in taluks_data.items():
                tb = tdata["bbox"]
                folium.Rectangle(
                    bounds=[[tb[2], tb[0]], [tb[3], tb[1]]],
                    color="#ed8936", weight=1.5, fill=True, fill_opacity=0.05,
                    tooltip=f"Taluk: {tname}",
                ).add_to(m)
        
        st_folium(m, use_container_width=True, height=520)
        st.info("👆 Select your area in the sidebar, then click **Compute Risk Scores** to see slope-level analysis.")
        
    else:
        df = st.session_state.slope_df
        
        # KPI row
        counts = df["alert_level"].value_counts().to_dict()
        total = len(df)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        kpis = [
            (col1, "⚫", counts.get("EXTREME", 0), "EXTREME", "#ff2929"),
            (col2, "🔴", counts.get("HIGH", 0), "HIGH", "#ff6b6b"),
            (col3, "🟠", counts.get("ELEVATED", 0), "ELEVATED", "#ffa500"),
            (col4, "🟡", counts.get("MODERATE", 0), "MODERATE", "#ffd700"),
            (col5, "🟢", counts.get("LOW", 0), "LOW", "#27ae60"),
            (col6, "📍", total, "TOTAL", "#667eea"),
        ]
        for col, emoji, val, label, color in kpis:
            col.markdown(f"""
            <div style="background:white;border-radius:10px;padding:0.6rem;text-align:center;
                       box-shadow:0 2px 6px rgba(0,0,0,0.08);border-top:3px solid {color};">
                <div style="font-size:1.6rem;font-weight:700;color:{color};">{val}</div>
                <div style="font-size:0.7rem;color:#718096;">{emoji} {label}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Alert banner
        extreme_count = counts.get("EXTREME", 0) + counts.get("HIGH", 0)
        if extreme_count > 0:
            st.markdown(f"""
            <div class="alert-extreme">
                ⚠️ <b>{extreme_count} slope units</b> at HIGH or EXTREME risk — immediate field verification recommended.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                ✅ No extreme risk slopes detected — continue monitoring.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # MAP
        center = get_current_center()
        zoom = get_zoom_for_level()
        m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, tiles="CartoDB positron")
        
        # Colour scale
        color_map = {
            "EXTREME": "#cc0000", "HIGH": "#ff4444",
            "ELEVATED": "#ff8c00", "MODERATE": "#ffd700", "LOW": "#2ecc71"
        }
        radius_map = {"EXTREME": 12, "HIGH": 10, "ELEVATED": 8, "MODERATE": 7, "LOW": 6}
        
        # Boundary box
        bbox = get_current_bbox()
        folium.Rectangle(
            bounds=[[bbox[2], bbox[0]], [bbox[3], bbox[1]]],
            color="#667eea", weight=1.5, fill=False,
        ).add_to(m)
        
        # Add slope markers
        for _, row in df.iterrows():
            lvl = row.get("alert_level", "LOW")
            c = color_map.get(lvl, "#888")
            r = radius_map.get(lvl, 6)
            
            popup_html = f"""
            <div style="font-family:sans-serif;width:240px;">
                <div style="background:{c};color:white;padding:6px 10px;border-radius:6px 6px 0 0;">
                    <b>{row['slope_id']}</b>
                    <span style="float:right;font-size:0.8rem;">{alert_emoji(lvl)} {lvl}</span>
                </div>
                <div style="padding:8px 10px;border:1px solid #eee;border-radius:0 0 6px 6px;">
                    <b>📍 Location</b><br>
                    State: {row.get('state','')}<br>
                    District: {row.get('district','')}<br>
                    Taluk: {row.get('taluk','')}<br><br>
                    <b>⚠️ Risk Score: {row.get('risk_score',0)}/100</b><br>
                    Factor of Safety: {row.get('FS','N/A')}<br>
                    FS Status: {row.get('status','N/A')}<br><br>
                    <b>🌧️ Rainfall</b><br>
                    Today: {row.get('R_24hr',0):.1f} mm<br>
                    7-day: {row.get('R_7day',0):.1f} mm<br>
                    SWI: {row.get('SWI',0):.2f}<br><br>
                    <b>⛰️ Terrain</b><br>
                    Slope: {row.get('slope_angle_deg',0):.1f}°<br>
                    Elevation: {row.get('elevation_m',0):.0f} m<br>
                    Soil: {row.get('soil_type','')}<br>
                    ID Exceeded: {'Yes ⚠️' if row.get('id_exceeded') else 'No ✅'}
                </div>
            </div>
            """
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=r,
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['slope_id']} | {lvl} | Score:{row.get('risk_score',0)}"
            ).add_to(m)
        
        # District/taluk boundaries overlay
        if not st.session_state.selected_district:
            for dname, ddata in ADMIN_HIERARCHY.get(st.session_state.selected_state, {}).get("districts", {}).items():
                db = ddata["bbox"]
                folium.Rectangle(
                    bounds=[[db[2], db[0]], [db[3], db[1]]],
                    color="#48bb78", weight=1, fill=False, dash_array="5",
                    tooltip=f"District: {dname}",
                ).add_to(m)
        
        if st.session_state.selected_district and not st.session_state.selected_taluk:
            from engine.admin_hierarchy import get_all_taluks_in_district
            taluks_data = get_all_taluks_in_district(st.session_state.selected_state, st.session_state.selected_district)
            for tname, tdata in taluks_data.items():
                tb = tdata["bbox"]
                folium.Rectangle(
                    bounds=[[tb[2], tb[0]], [tb[3], tb[1]]],
                    color="#ed8936", weight=1, fill=False, dash_array="3",
                    tooltip=f"Taluk: {tname}",
                ).add_to(m)
        
        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;right:30px;z-index:1000;background:white;
                    padding:10px 14px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
                    font-family:sans-serif;font-size:12px;">
            <b>Risk Levels</b><br>
            ⚫ EXTREME (&gt;80)<br>
            🔴 HIGH (60–80)<br>
            🟠 ELEVATED (40–60)<br>
            🟡 MODERATE (20–40)<br>
            🟢 LOW (&lt;20)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        st_folium(m, use_container_width=True, height=540)
        
        # Quick stats below map
        if st.session_state.selected_district:
            st.markdown("**Breakdown by Taluk:**")
            taluk_summary = df.groupby("taluk").agg(
                Total=("slope_id", "count"),
                Extreme=("alert_level", lambda x: (x == "EXTREME").sum()),
                High=("alert_level", lambda x: (x == "HIGH").sum()),
                Avg_Score=("risk_score", "mean"),
                Avg_FS=("FS", "mean"),
            ).round(2).reset_index()
            st.dataframe(taluk_summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: SLOPE DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    if not st.session_state.computed or st.session_state.slope_df is None:
        st.info("Compute risk scores first (sidebar button).")
    else:
        df = st.session_state.slope_df
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        filter_level = col1.multiselect(
            "Alert Level", ["EXTREME", "HIGH", "ELEVATED", "MODERATE", "LOW"],
            default=["EXTREME", "HIGH", "ELEVATED"]
        )
        if st.session_state.selected_district:
            taluk_list = sorted(df["taluk"].unique().tolist())
            filter_taluk = col2.multiselect("Taluk", taluk_list, default=taluk_list[:3] if len(taluk_list) > 3 else taluk_list)
        else:
            district_list = sorted(df["district"].unique().tolist())
            filter_taluk = col2.multiselect("District", district_list, default=district_list[:3] if len(district_list) > 3 else district_list)
        
        min_score = col3.slider("Min Risk Score", 0, 100, 0)
        sort_by = col4.selectbox("Sort By", ["risk_score", "FS", "R_24hr", "slope_angle_deg"])
        
        # Apply filters
        filtered = df[df["alert_level"].isin(filter_level) if filter_level else df["alert_level"].notna()]
        filtered = filtered[filtered["risk_score"] >= min_score]
        
        if st.session_state.selected_district and filter_taluk:
            filtered = filtered[filtered["taluk"].isin(filter_taluk)]
        elif not st.session_state.selected_district and filter_taluk:
            filtered = filtered[filtered["district"].isin(filter_taluk)]
        
        filtered = filtered.sort_values(sort_by, ascending=(sort_by == "FS"))
        
        st.markdown(f"**Showing {len(filtered)} of {len(df)} slope units**")
        
        # Slope cards
        if len(filtered) == 0:
            st.warning("No slopes match the selected filters.")
        else:
            for i, (_, row) in enumerate(filtered.head(30).iterrows()):
                lvl = row.get("alert_level", "LOW")
                c = alert_color(lvl)
                em = alert_emoji(lvl)
                fs = row.get("FS", 0)
                
                with st.container():
                    st.markdown(f"""
                    <div class="slope-card" style="border-color:{c};">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <b style="color:{c};">{em} {row['slope_id']}</b>
                                <span style="margin-left:10px;color:#4a5568;font-size:0.85rem;">
                                    📍 {row.get('state','')} › {row.get('district','')} › {row.get('taluk','')}
                                </span>
                            </div>
                            <div style="text-align:right;">
                                <span style="background:{c};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;">{lvl}</span>
                            </div>
                        </div>
                        <div style="display:flex;gap:2rem;margin-top:6px;font-size:0.82rem;color:#4a5568;">
                            <span>🎯 Score: <b>{row.get('risk_score',0)}</b>/100</span>
                            <span>⚖️ FS: <b style="color:{'#e53e3e' if fs<1.2 else '#276749'}">{fs}</b></span>
                            <span>🌧️ Today: <b>{row.get('R_24hr',0):.1f}mm</b></span>
                            <span>💧 SWI: <b>{row.get('SWI',0):.2f}</b></span>
                            <span>⛰️ {row.get('slope_angle_deg',0):.1f}°</span>
                            <span>🪨 {row.get('soil_type','')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(filtered) > 30:
                st.caption(f"Showing top 30 of {len(filtered)} results. Export CSV below to see all.")
        
        st.markdown("---")
        
        # District/Taluk comparison chart
        st.subheader("📊 Risk Distribution by Area")
        
        if st.session_state.selected_district:
            group_col = "taluk"
        else:
            group_col = "district"
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Risk score by area (box plot)
            fig = px.box(
                df, x=group_col, y="risk_score",
                color=group_col,
                title=f"Risk Score Distribution by {group_col.title()}",
                labels={"risk_score": "Risk Score", group_col: group_col.title()},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            # FS histogram
            fig2 = px.histogram(
                df, x="FS", color="alert_level",
                title="Factor of Safety Distribution",
                color_discrete_map={
                    "EXTREME": "#cc0000", "HIGH": "#ff4444",
                    "ELEVATED": "#ff8c00", "MODERATE": "#ffd700", "LOW": "#2ecc71"
                },
                nbins=30,
            )
            fig2.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="FS=1.0 (Failure)")
            fig2.add_vline(x=1.5, line_dash="dot", line_color="orange", annotation_text="FS=1.5")
            fig2.update_layout(height=300)
            col1.plotly_chart(fig2, use_container_width=True)
            
            # Slope angle vs risk scatter
            fig3 = px.scatter(
                df, x="slope_angle_deg", y="risk_score",
                color="alert_level", size="R_24hr",
                color_discrete_map={
                    "EXTREME": "#cc0000", "HIGH": "#ff4444",
                    "ELEVATED": "#ff8c00", "MODERATE": "#ffd700", "LOW": "#2ecc71"
                },
                title="Slope Angle vs Risk Score",
                labels={"slope_angle_deg": "Slope Angle (°)", "risk_score": "Risk Score"},
                hover_data=["slope_id", group_col, "FS"],
            )
            fig3.update_layout(height=300)
            col2.plotly_chart(fig3, use_container_width=True)
            
        except ImportError:
            # Fallback to streamlit native charts
            st.bar_chart(df.groupby(group_col)["risk_score"].mean())
        
        # Export
        st.markdown("---")
        csv = filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Export Filtered Data as CSV",
            csv,
            f"hlpe_risk_{st.session_state.selected_state}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 3: RAINFALL
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🌧️ Rainfall Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Add Manual Station Data**")
        station_name = st.text_input("Station Name", "")
        manual_lat = st.number_input("Latitude", value=get_current_center()[0], format="%.4f")
        manual_lon = st.number_input("Longitude", value=get_current_center()[1], format="%.4f")
        today_rain = st.number_input("Today's Rainfall (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5)
        yesterday_rain = st.number_input("Yesterday's Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
        
        if st.button("➕ Add Station", use_container_width=True):
            if "rain_stations" not in st.session_state:
                st.session_state.rain_stations = []
            st.session_state.rain_stations.append({
                "name": station_name or f"Station_{len(st.session_state.get('rain_stations',[]))+1}",
                "lat": manual_lat, "lon": manual_lon,
                "today_mm": today_rain, "yesterday_mm": yesterday_rain,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.manual_rainfall = {"today_mm": today_rain}
            st.success("Station added!")
        
        # Show station list
        if "rain_stations" in st.session_state and st.session_state.rain_stations:
            st.markdown("**Active Stations:**")
            for s in st.session_state.rain_stations:
                st.markdown(f"📡 **{s['name']}** — {s['today_mm']} mm today")
    
    with col2:
        if st.session_state.computed and st.session_state.slope_df is not None:
            df = st.session_state.slope_df
            
            try:
                import plotly.express as px
                
                group_col = "taluk" if st.session_state.selected_district else "district"
                
                # Rainfall by area
                rain_summary = df.groupby(group_col).agg(
                    R_24hr=("R_24hr", "mean"),
                    R_7day=("R_7day", "mean"),
                    R_15day=("R_15day", "mean"),
                    SWI=("SWI", "mean"),
                ).round(2).reset_index()
                
                fig = px.bar(
                    rain_summary.melt(id_vars=group_col, value_vars=["R_24hr", "R_7day", "R_15day"]),
                    x=group_col, y="value", color="variable", barmode="group",
                    title=f"Rainfall by {group_col.title()}",
                    labels={"value": "Rainfall (mm)", group_col: group_col.title(), "variable": "Period"},
                    color_discrete_map={"R_24hr": "#3182ce", "R_7day": "#805ad5", "R_15day": "#d53f8c"},
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                # SWI vs Risk scatter
                fig2 = px.scatter(
                    df, x="SWI", y="risk_score",
                    color="alert_level", title="Soil Water Index vs Risk Score",
                    labels={"SWI": "Soil Water Index", "risk_score": "Risk Score"},
                    color_discrete_map={
                        "EXTREME": "#cc0000", "HIGH": "#ff4444",
                        "ELEVATED": "#ff8c00", "MODERATE": "#ffd700", "LOW": "#2ecc71"
                    },
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
                
            except ImportError:
                st.info("Install plotly for charts: pip install plotly")
                st.dataframe(df[["slope_id", "taluk", "R_24hr", "R_7day", "SWI"]].head(20))
        else:
            st.info("Run computation first to see rainfall analysis.")
            
            # ID Threshold demo
            st.markdown("**Intensity-Duration Threshold Curve (Preview)**")
            try:
                import plotly.graph_objects as go
                
                durations = np.linspace(0.1, 72, 100)
                alpha, beta = 10.0, 0.40
                threshold = alpha * durations ** (-beta)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=durations, y=threshold, mode='lines', name='I-D Threshold',
                                        line=dict(color='red', width=2)))
                fig.update_layout(
                    title="Rainfall Intensity-Duration Threshold",
                    xaxis_title="Duration (hours)", yaxis_title="Intensity (mm/hr)",
                    height=280, xaxis_type="log", yaxis_type="log",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass


# ═══════════════════════════════════════════════════════════════════
# TAB 4: UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📁 Upload Local Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Slope Unit CSV**")
        st.caption("Required columns: `lat`, `lon`, `slope_angle_deg`, `soil_type`. Optional: `state`, `district`, `taluk`")
        
        uploaded_csv = st.file_uploader("Slope Units CSV", type=["csv"], key="slope_csv")
        if uploaded_csv:
            try:
                user_df = pd.read_csv(uploaded_csv)
                st.success(f"✅ Loaded {len(user_df)} slope units")
                st.dataframe(user_df.head(5), use_container_width=True)
                st.session_state.uploaded_df = user_df
                st.session_state.computed = False
                st.info("Click 'Compute Risk Scores' in sidebar to process your data.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        
        st.markdown("---")
        st.markdown("**Upload Landslide Inventory CSV**")
        st.caption("Columns: `lat`, `lon`, `date`, `type`, `district`, `taluk`")
        
        inventory_csv = st.file_uploader("Landslide Inventory", type=["csv"], key="inventory_csv")
        if inventory_csv:
            try:
                inv_df = pd.read_csv(inventory_csv)
                st.success(f"✅ Loaded {len(inv_df)} past events")
                st.dataframe(inv_df.head(5), use_container_width=True)
                if "inventory_df" not in st.session_state:
                    st.session_state.inventory_df = inv_df
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("**Upload Shapefile / GeoJSON**")
        st.caption("For slope unit polygons, geology layers, or administrative boundaries")
        
        geojson_file = st.file_uploader("GeoJSON File", type=["geojson", "json"], key="geojson")
        if geojson_file:
            try:
                gj = json.load(geojson_file)
                feature_count = len(gj.get("features", []))
                st.success(f"✅ Loaded {feature_count} features")
                
                # Show on map
                center = get_current_center()
                m = folium.Map(location=[center[0], center[1]], zoom_start=get_zoom_for_level())
                folium.GeoJson(gj, name="Uploaded Layer", tooltip=folium.GeoJsonTooltip(fields=list(
                    gj["features"][0]["properties"].keys())[:3] if gj.get("features") else []
                )).add_to(m)
                st_folium(m, use_container_width=True, height=350)
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("**Log New Landslide Event**")
        with st.form("new_event_form"):
            ev_lat = st.number_input("Latitude", value=get_current_center()[0], format="%.5f")
            ev_lon = st.number_input("Longitude", value=get_current_center()[1], format="%.5f")
            ev_date = st.date_input("Date", value=datetime.now())
            ev_type = st.selectbox("Type", ["Shallow Slip", "Debris Flow", "Rockfall", "Mudflow", "Deep Seated"])
            ev_notes = st.text_area("Notes / Description", height=80)
            ev_submit = st.form_submit_button("➕ Add Event", type="primary")
            if ev_submit:
                if "event_log" not in st.session_state:
                    st.session_state.event_log = []
                st.session_state.event_log.append({
                    "lat": ev_lat, "lon": ev_lon, "date": str(ev_date),
                    "type": ev_type, "notes": ev_notes,
                    "state": st.session_state.selected_state,
                    "district": st.session_state.selected_district or "",
                    "taluk": st.session_state.selected_taluk or "",
                })
                st.success(f"✅ Event logged: {ev_type} on {ev_date}")
        
        if "event_log" in st.session_state and st.session_state.event_log:
            st.markdown(f"**{len(st.session_state.event_log)} events logged this session**")
            ev_csv = pd.DataFrame(st.session_state.event_log).to_csv(index=False)
            st.download_button("⬇️ Export Event Log", ev_csv, "landslide_events.csv", "text/csv")
    
    # CSV template download
    st.markdown("---")
    st.markdown("**Download CSV Templates**")
    col1, col2 = st.columns(2)
    
    template_slope = pd.DataFrame({
        "slope_id": ["SU_0001", "SU_0002"],
        "state": ["Kerala", "Kerala"],
        "district": ["Idukki", "Wayanad"],
        "taluk": ["Devikulam", "Mananthavady"],
        "lat": [10.15, 11.85],
        "lon": [77.2, 76.1],
        "elevation_m": [800, 950],
        "slope_angle_deg": [28.5, 35.2],
        "soil_type": ["Laterite", "Sandy Loam"],
        "soil_depth_m": [1.5, 2.0],
        "cohesion_kpa": [8.0, 5.0],
        "friction_angle_deg": [28.0, 32.0],
    })
    col1.download_button(
        "⬇️ Slope Units Template",
        template_slope.to_csv(index=False),
        "slope_units_template.csv", "text/csv", use_container_width=True
    )
    
    template_inv = pd.DataFrame({
        "lat": [10.15, 11.85],
        "lon": [77.2, 76.1],
        "date": ["2024-07-15", "2023-08-02"],
        "type": ["Shallow Slip", "Debris Flow"],
        "state": ["Kerala", "Kerala"],
        "district": ["Idukki", "Wayanad"],
        "taluk": ["Devikulam", "Mananthavady"],
        "description": ["Road cut failure", "Forest slope failure"],
    })
    col2.download_button(
        "⬇️ Inventory Template",
        template_inv.to_csv(index=False),
        "landslide_inventory_template.csv", "text/csv", use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════
# TAB 5: REPORT
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📄 Daily Risk Report")
    
    if not st.session_state.computed or st.session_state.slope_df is None:
        st.info("Compute risk scores first.")
    else:
        df = st.session_state.slope_df
        
        # Report header
        date_str = datetime.now().strftime("%B %d, %Y | %H:%M IST")
        area_str = f"{st.session_state.selected_state}"
        if st.session_state.selected_district:
            area_str += f" › {st.session_state.selected_district}"
        if st.session_state.selected_taluk:
            area_str += f" › {st.session_state.selected_taluk}"
        
        counts = df["alert_level"].value_counts().to_dict()
        
        report_md = f"""
# 🏔️ HLPE — Daily Landslide Risk Report

**Area:** {area_str}  
**Date:** {date_str}  
**Total Slope Units Analysed:** {len(df)}

---

## Executive Summary

| Alert Level | Count | % |
|-------------|-------|---|
| ⚫ EXTREME | {counts.get('EXTREME', 0)} | {counts.get('EXTREME', 0)/len(df)*100:.1f}% |
| 🔴 HIGH | {counts.get('HIGH', 0)} | {counts.get('HIGH', 0)/len(df)*100:.1f}% |
| 🟠 ELEVATED | {counts.get('ELEVATED', 0)} | {counts.get('ELEVATED', 0)/len(df)*100:.1f}% |
| 🟡 MODERATE | {counts.get('MODERATE', 0)} | {counts.get('MODERATE', 0)/len(df)*100:.1f}% |
| 🟢 LOW | {counts.get('LOW', 0)} | {counts.get('LOW', 0)/len(df)*100:.1f}% |

**Average Factor of Safety:** {df['FS'].mean():.3f}  
**Average Today's Rainfall:** {df['R_24hr'].mean():.1f} mm  
**Average 7-day Rainfall:** {df['R_7day'].mean():.1f} mm

---

## 🚨 Priority Slopes (Extreme + High)

"""
        priority = df[df["alert_level"].isin(["EXTREME", "HIGH"])].sort_values("risk_score", ascending=False).head(10)
        if len(priority) > 0:
            for _, row in priority.iterrows():
                report_md += f"""
**{row['slope_id']}** — {alert_emoji(row['alert_level'])} {row['alert_level']}  
📍 {row.get('district','')} › {row.get('taluk','')} | Lat: {row['lat']}, Lon: {row['lon']}  
Risk Score: {row['risk_score']}/100 | FS: {row['FS']} | Today: {row['R_24hr']:.1f}mm | Slope: {row['slope_angle_deg']:.1f}°  

"""
        else:
            report_md += "_No EXTREME or HIGH risk slopes at this time._\n"
        
        report_md += f"""
---

## Area-Level Summary

"""
        group_col = "taluk" if st.session_state.selected_district else "district"
        area_sum = df.groupby(group_col).agg(
            Units=("slope_id", "count"),
            Extreme=("alert_level", lambda x: (x == "EXTREME").sum()),
            High=("alert_level", lambda x: (x == "HIGH").sum()),
            Avg_Score=("risk_score", "mean"),
            Avg_FS=("FS", "mean"),
            Rain_Today=("R_24hr", "mean"),
        ).round(2).reset_index()
        
        report_md += area_sum.to_markdown(index=False)
        report_md += f"""

---

*Generated by HLPE — Hybrid Landslide Prediction Engine*  
*Model: Infinite Slope (FS) + I-D Threshold + SWI Consensus*  
*Data: Open-Meteo rainfall | Synthetic DEM | HLPE Physics Engine*
"""
        
        st.markdown(report_md)
        
        st.download_button(
            "⬇️ Download Report (Markdown)",
            report_md,
            f"HLPE_Report_{st.session_state.selected_state}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            "text/markdown",
            type="primary",
            use_container_width=True,
        )
