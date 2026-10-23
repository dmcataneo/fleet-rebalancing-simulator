### The problem

Shared vehicles drift. Trips flow from homes to offices in the morning and back in the evening, so some zones run empty while others fill up. A rider who finds no vehicle is lost demand. Moving vehicles costs money, so the question is how much to move, where and when.

### What the model does

1. A city of 12 x 12 zones with residential, business and leisure areas.
2. Trip starts per zone and hour follow commute and leisure patterns. Destinations follow a gravity rule: attractive and close zones get more trips.
3. Each hour a trip is served only if a vehicle is in the zone. Otherwise it is lost.
4. Three policies:
   - **none**: leave the fleet where riders put it.
   - **night_reset**: once a night, move vehicles toward targets based on the expected next-day demand.
   - **hourly_reactive**: every hour, move up to a budget of vehicles toward targets based on expected demand over the next few hours.

### Ideas behind it

- **Transportation problem.** Turning current stock into target stock at least distance is a minimum-cost flow (Hitchcock, 1941). The app solves it as a linear program. You can switch to the Hungarian method or a nearest-first rule to compare.
- **Base-stock targets.** Target stock per zone is proportional to expected demand. Simple and readable. It does not add extra buffer for volatile zones, which a newsvendor-style target would.
- **Overnight versus intraday moves.** The night reset moves vehicles when few are in use. The hourly policy keeps adjusting during the day. The research on station-based bike sharing treats repositioning with trucks as a routing problem, which this project leaves out (Raviv, Tzur and Forma, 2013; Schuijbroek, Hampshire and van Hoeve, 2017).

### Is the Hungarian algorithm a good fit?

Partly. The Hungarian method solves the **assignment problem**: match n items to n slots one-to-one at minimum total cost. Rebalancing is a **transportation problem** (zones hold many vehicles). You can still use it by expanding each zone into single vehicles and single empty slots. The app does that ("hungarian" matching) and the tests check it reaches the same minimum distance as the linear program. Nearest-first matching moves the same vehicles but is not guaranteed to find the shortest total distance. The matching methods can be switched in the sidebar. Within one rebalancing step, LP and Hungarian reach the same minimum distance. Over a whole simulation the totals still differ a little, because ties are broken differently and the fleet then follows a different path, and a policy that is cheapest in one step is not guaranteed to be cheapest or best over a day.

So the answer is: correct, but not the best tool here. The LP is smaller and more natural, and the Hungarian version only wins on convenience. The method fits better when the question is truly one-to-one, for example matching specific idle vehicles, drivers or couriers to specific tasks at one moment. That is a different project from fleet balance, and it is not forced into this one.

### What the model does not do

- No truck routes or capacity. Cost is a flat amount per vehicle-km.
- Every trip takes one hour. No battery, repairs, theft or weather.
- Riders do not walk to the next zone and demand does not react to availability. This probably overstates how much a missing vehicle costs and understates the effect of low availability on future demand.
- Demand shapes are assumptions, not measurements. The numbers show how the model behaves, not what a real operator would see.
- The papers above study station-based bike sharing with trucks. This model uses zones and a flat cost per vehicle-km, so it is a simplified cousin, not a replication.
- It models vehicles in zones, not ride-hail drivers or couriers. Those share the idea of matching supply to demand in space and time, but dispatch, pricing and driver behaviour differ, and none of that is modelled here.

### References

- Raviv, T., Tzur, M., Forma, I. A. (2013). Static repositioning in a bike-sharing system: models and solution approaches. EURO Journal on Transportation and Logistics. https://doi.org/10.1007/s13676-012-0017-6
- Schuijbroek, J., Hampshire, R. C., van Hoeve, W.-J. (2017). Inventory rebalancing and vehicle routing in bike sharing systems. European Journal of Operational Research. https://doi.org/10.1016/j.ejor.2016.08.029
- Hitchcock, F. L. (1941). The distribution of a product from several sources to numerous localities. Journal of Mathematics and Physics, 20. https://doi.org/10.1002/sapm1941201224
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2. https://doi.org/10.1002/nav.3800020109
