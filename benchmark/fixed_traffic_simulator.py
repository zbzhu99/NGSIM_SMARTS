from traffic_simulator import TrafficSim


class FixedTrafficSim(TrafficSim):
    def __init__(self, vehicle_ids, *args, **kwargs):
        self.vehicle_ids = vehicle_ids
        super().__init__(*args, **kwargs)

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_itr = 0
