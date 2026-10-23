"""Streamlit app: watch a fictional shared fleet and test rebalancing policies."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fleet_balance.city import make_city
from fleet_balance.sim import (POLICIES, Config, compare, cost_per_extra_trip,
                               hourly_profile, run)

st.set_page_config(page_title="Fleet Balance", layout="wide")
st.title("Fleet Balance")
st.caption("Shared bikes and scooters in a fictional city. All data is simulated.")

with st.sidebar:
    st.header("Scenario")
    fleet = st.slider("Fleet size", 400, 3000, 1500, 100)
    trips = st.slider("Trip requests per day", 2000, 12000, 6000, 500)
    weekend = st.checkbox("Weekend pattern", False)
    days = st.slider("Days simulated", 2, 10, 4)
    st.header("Rebalancing")
    hourly_budget = st.slider("Reactive: vehicles movable per hour", 0, 200, 60, 10)
    night_budget = st.slider("Night reset: vehicles movable", 0, 1500, 600, 50)
    lookahead = st.slider("Reactive: look-ahead hours", 1, 6, 3)
    cost_km = st.number_input("Cost per vehicle-km moved (fictional)", 0.0, 10.0, 0.9, 0.1)
    matcher = st.selectbox("Matching method", ["lp", "hungarian", "greedy"],
                           help="lp and hungarian give the same optimal distance. greedy is nearest-first.")
    seed = st.number_input("Random seed", 0, 9999, 7)

city = make_city(12)
cfg = Config(fleet=fleet, trips_per_day=trips, days=days, weekend=weekend,
             hourly_budget=hourly_budget, night_budget=night_budget,
             lookahead_hours=lookahead, cost_per_vehicle_km=cost_km, matcher=matcher, seed=int(seed))


@st.cache_data(show_spinner="Simulating...")
def cached_run(cfg_dict):
    return run(city, Config(**cfg_dict))


@st.cache_data(show_spinner="Comparing policies on 5 seeds...")
def cached_compare(cfg_dict):
    return compare(city, Config(**cfg_dict), seeds=5)


tab_map, tab_cmp, tab_theory = st.tabs(["Heatmap", "Compare policies", "Theory and limits"])

METRICS = {
    "Lost trips (no vehicle in zone)": ("lost", "Reds"),
    "Vehicles available at start of hour": ("stock", "Blues"),
    "Trip requests": ("demand", "Oranges"),
    "Vehicles minus requests (blue = spare, red = short)": ("gap", "RdBu"),
}


def grid_for(res, key, hour, whole_day):
    n = city.n
    arr = {"lost": res.lost, "stock": res.stock, "demand": res.demand,
           "gap": res.stock - res.demand}[key].reshape(days, 24, -1).mean(axis=0)
    g = arr.sum(axis=0) if (whole_day and key in ("lost", "demand")) else arr[hour]
    return g.reshape(n, n)


with tab_map:
    st.caption("Zones of a made-up 12 x 12 grid, not a real city map. Values are averages over the simulated days. "
               "Vehicles available is counted at the start of the hour, after arrivals and rebalancing, before that hour's trips.")
    c1, c2, c3, c4 = st.columns([2, 3, 3, 2])
    policy = c1.selectbox("Policy", POLICIES, index=2)
    metric = c2.selectbox("Show", list(METRICS))
    hour = c3.slider("Hour of day", 0, 23, 18)
    side = c4.checkbox("Compare with 'none'", True)
    key, scale = METRICS[metric]
    whole_day = key in ("lost", "demand") and st.checkbox("Sum over the whole day instead of one hour", False)
    shown = ["none", policy] if (side and policy != "none") else [policy]
    results = {p: cached_run({**cfg.__dict__, "policy": p}) for p in shown}
    grids = {p: grid_for(results[p], key, hour, whole_day) for p in shown}
    if key == "gap":
        lim = max(float(np.abs(g).max()) for g in grids.values()) or 1.0
        zr = (-lim, lim)
    else:
        zr = (0.0, max(float(g.max()) for g in grids.values()) or 1.0)
    cols = st.columns(len(shown))
    when = "whole day" if whole_day else f"{hour:02d}:00"
    for col, p in zip(cols, shown):
        fig = px.imshow(grids[p], color_continuous_scale=scale, zmin=zr[0], zmax=zr[1], aspect="equal")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=40, b=0),
                          title=f"{p}: {when}", coloraxis_colorbar=dict(title="trips" if key in ("lost", "demand") else "vehicles"))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        col.plotly_chart(fig, width="stretch")
        col.metric(f"Trips served, all hours and days ({p})", f"{results[p].served_share:.1%}",
                   help="Whole run, this random seed only. It does not change with the hour slider. The Compare tab shows the 5-seed average, so the two can differ by about a point.")
    st.caption("Both maps use the same colour scale so they can be compared directly.")
    r0 = results[shown[-1]]
    prof = hourly_profile(r0, days)
    f2 = go.Figure()
    f2.add_bar(x=list(range(24)), y=prof["demand"], name="Requests")
    f2.add_bar(x=list(range(24)), y=prof["lost"], name="Lost", marker_color="crimson")
    f2.add_vline(x=hour, line_dash="dot")
    f2.update_layout(barmode="overlay", height=300, title="Requests and lost trips",
                     xaxis_title="hour of day", yaxis_title="trips per hour (city total)",
                     margin=dict(l=0, r=0, t=40, b=0))
    st.caption(f"Hourly profile for policy: {shown[-1]}")
    st.plotly_chart(f2, width="stretch")

with tab_cmp:
    m = cached_compare(cfg.__dict__)
    extra = cost_per_extra_trip(m)
    rows = []
    for p in POLICIES:
        rows.append({"policy": p, "trips served": f"{m[p]['served_share']:.1%}",
                     "lost trips / day": round(m[p]["lost_trips_per_day"]),
                     "vehicles moved / day": round(m[p]["moved_vehicles_per_day"]),
                     "move cost / day": round(m[p]["move_cost_per_day"]),
                     "cost per extra trip served": "-" if extra[p] is None else f"{extra[p]:.2f}"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Mean of 5 random seeds. Cost per extra trip compares with doing nothing. "
               "It says nothing about revenue: the value of a served trip is not modelled.")
    fig = px.bar(x=POLICIES, y=[m[p]["lost_trips_per_day"] for p in POLICIES],
                 labels={"x": "policy", "y": "lost trips per day"})
    st.plotly_chart(fig, width="stretch")
    st.info("Try a smaller and a bigger fleet and watch how the gap between policies changes. "
            "Then compare the cost of moving vehicles with what the extra trips are worth to you. "
            "That value is not part of the model.")

with tab_theory:
    st.markdown((__import__("pathlib").Path(__file__).parent / "THEORY.md").read_text())


st.divider()
st.caption("By [dmcataneo](https://github.com/dmcataneo) · "
           "[Source code](https://github.com/dmcataneo/fleet-rebalancing-simulator)")
