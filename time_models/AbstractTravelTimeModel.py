import logging
from abc import ABC, abstractmethod
from time_models.basic_informations import LIST_OF_STATIONS


class AbstractTravelTimeModel(ABC):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('TravelTimeModel')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    '''
    def config_logger(self, use_console: bool = False, file_path: str | None = None, reset_handlers: bool = False):
        standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if use_console:
            ch = logging.StreamHandler()
            ch.setFormatter(standard_formatter)
            self.logger.addHandler(ch)
        if file_path is not None:
            fh = logging.FileHandler(file_path)
            fh.setFormatter(standard_formatter)
            self.logger.addHandler(fh)
        if reset_handlers:
            self.logger.handlers = []
    '''
    def travel_time(self, origin_station, destination: []) -> float:
        """
        Returns the estimated time to travel from the origin_station to the destination in seconds
        :param origin_station: The station from which the ambulance is dispatched, either index (0..17) or station name
        :param destination: The destination, as an array [lng, lat]
        :return: The estimated time to travel from the origin_station to the destination in seconds
        """
        if isinstance(origin_station, str):
            origin_station = LIST_OF_STATIONS[origin_station]['id']

        if not isinstance(origin_station, int):
            raise TypeError("origin_station must be an int or a str")

        return self._travel_time(origin_station, destination)

    @abstractmethod
    def _travel_time(self, origin_station: int, destination: []) -> float:
        raise NotImplementedError('Implement this method in your subclass')

    @abstractmethod
    def _travel_times(self, destination: tuple) -> []:
        """
        Returns the estimated time to travel from each station to the destination in seconds
        :param destination: The destination, as an array [lng, lat]
        :return: The estimated time to travel from every station to the destination in seconds
        """
        raise NotImplementedError('Implement this method in your subclass')
