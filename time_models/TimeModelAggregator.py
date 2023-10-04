from AbstractTravelTimeModel import AbstractTravelTimeModel
from AbstractOnSiteTimeModel import AbstractOnSiteTimeModel


class TimeModelAggregator:
    def __init__(self, travel_time_model: AbstractTravelTimeModel, on_site_model: AbstractOnSiteTimeModel):
        self.travel_time_model = travel_time_model
        self.on_site_model = on_site_model

    def config_logger(self, use_console: bool = False, file_path: str | None = None):
        self.travel_time_model.config_logger(use_console, file_path)
        self.on_site_model.config_logger(use_console, file_path)

    def get_travel_time(self, origin, destination) -> float:
        return self.travel_time_model.travel_time(origin, destination)

    def get_on_site_time(self, **kwargs) -> float:
        return self.on_site_model.on_site_time(**kwargs)

    def get_full_mission_time(self, origin, destination, **kwargs) -> float:
        return self.get_travel_time(origin, destination) * 2 + self.get_on_site_time(**kwargs)
