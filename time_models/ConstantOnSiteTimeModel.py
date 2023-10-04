import logging
from AbstractOnSiteTimeModel import AbstractOnSiteTimeModel


class ConstantOnSiteTimeModel(AbstractOnSiteTimeModel):
    def __init__(self, default_time: float = 3600.0):
        super().__init__()
        self.default_time = default_time

    def on_site_time(self, **kwargs) -> float:
        """
        Returns the estimated time to be on site in seconds.

        **Keyword Arguments**\n
        * **time** (`float`) --
          Time to return in seconds if specified. Else, returns the default, 1 hour.

        :param kwargs: Keyword arguments, see above.

        :return: Constant time in seconds
        """
        return kwargs.get('time', self.default_time)


def test_class():
    tm = ConstantOnSiteTimeModel()
    print(tm.on_site_time())
    print(tm.on_site_time(time=10))


if __name__ == '__main__':
    test_class()
