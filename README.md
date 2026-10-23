# Fleet Balance

A simulation of shared bikes and scooters on a fictional city, built to study one question: how much should an operator move vehicles, and when?

Trips start and end in zones with commute and leisure patterns. A trip is lost when no vehicle is in the zone. You can compare three policies (no rebalancing, a night reset, hourly reactive moves), look at where supply and demand diverge on a heatmap, and see the cost per extra trip served.

All data is generated. Nothing here comes from a real operator, and the results show how the model behaves, not what a real fleet would do. `THEORY.md` lists the ideas used (transportation problem, base-stock targets, Hungarian assignment) and what the model leaves out.

## Run

    pip install -r requirements.txt
    streamlit run app.py
    pytest

## Layout

- `fleet_balance/city.py` zones and distances
- `fleet_balance/demand.py` hourly demand and destination model
- `fleet_balance/rebalance.py` targets and three matching methods
- `fleet_balance/sim.py` hourly simulation and policy comparison
- `app.py` Streamlit interface
