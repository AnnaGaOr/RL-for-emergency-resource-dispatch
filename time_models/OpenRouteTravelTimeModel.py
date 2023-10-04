import requests

from time_models.AbstractTravelTimeModel import AbstractTravelTimeModel
from time_models.basic_informations import LIST_OF_STATIONS

# Parameters of the local server set in the lab. Change if needed.
LOCAL_SERVICE_IP = '10.192.67.146'
LOCAL_SERVICE_PORT = 8080
LOCAL_SERVICE_API_URL = f"http://{LOCAL_SERVICE_IP}:{LOCAL_SERVICE_PORT}/ors/v2/matrix/driving-car"

OPEN_ROUTE_SERVICE_API_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"


class OpenRouteTravelTimeModel(AbstractTravelTimeModel):
    def __init__(self, mode: str = 'local', api_url: str = None, api_key: str = None):
        """
        :param mode: 'local' to use the local server (at the lab, or via VPN only), online to use the online API (limited to 2500 requests per day, iirc)
        :param api_url: The mode is ignored if this parameter is set. The URL of the API to use.
        :param api_key: API key to use. Not needed if using the local server.
        """
        super().__init__()

        if api_url is not None:
            self.api_url = api_url
        else:
            if mode == 'local':
                self.api_url = LOCAL_SERVICE_API_URL
            elif mode == 'online':
                self.api_url = OPEN_ROUTE_SERVICE_API_URL
            else:
                raise ValueError(f'Unknown mode: {mode}')

        self.api_key = api_key

        self.base_locations = [[station['lng'], station['lat']] for station in LIST_OF_STATIONS.values()]

    def _travel_time(self, origin_station: int, destination: []) -> float:
        return self._travel_times(destination)[origin_station]

    def _travel_times(self, destination: []) -> []:
        self.logger.info(f'Requesting travel times from each station to destination {destination}...')
        response = requests.post(
            url=self.api_url,
            headers={
                'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                'Authorization': self.api_key,
                'Content-Type': 'application/json; charset=utf-8'
            },
            json={'locations': self.base_locations + [destination]}
        )
        self.logger.info('Received response from OpenRouteService.')
        return response.json()['durations'][-1]

    def is_location_valid(self, location) -> bool:
        times = self._travel_times(location)
        return not all(t is None for t in times)


def test_class():
    tm = OpenRouteTravelTimeModel(mode='local')
    #print(tm._travel_times([6.6335973, 46.5196535]))
    #print(tm.travel_time('Lausanne', [6.6335973, 46.5196535]))
    #print(tm.travel_time(1, [6.6335973, 46.5196535]))
    print(tm.travel_time(1, [6.2078656, 47.6643995]))

    print("FINISHED TESTING OpenRouteTravelTimeModel")


if __name__ == '__main__':
    test_class()
