import logging
from time_models.AbstractOnSiteTimeModel import AbstractOnSiteTimeModel
from numpy.random import normal


class OnSiteTimeModel(AbstractOnSiteTimeModel):
    def __init__(self, mean=4338.858777303995, std=2193.946985769463):
        super().__init__()
        self.mean = mean
        self.std = std

    def on_site_time(self, **kwargs) -> float:
        """
        Returns the estimated time to be on site in seconds.

        **Keyword Arguments**\n
        * **time** (`float`) --
          Time to return in seconds if specified. Else, returns the default, 1 hour.

        :param kwargs: Keyword arguments, see above.

        :return: Constant time in seconds
        """
        return normal(self.mean, self.std)


def test_class():
    tm = OnSiteTimeModel()
    print(tm.on_site_time())


if __name__ == '__main__':
    test_class()
