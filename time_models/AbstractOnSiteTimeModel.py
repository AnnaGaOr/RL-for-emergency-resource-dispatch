import logging
from abc import ABC, abstractmethod


class AbstractOnSiteTimeModel(ABC):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('OnSiteTimeModel')
        self.logger.setLevel(logging.INFO)

    '''
    def config_logger(self, use_console: bool = False, file_path: str | None = None):
        standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if use_console:
            ch = logging.StreamHandler()
            ch.setFormatter(standard_formatter)
            self.logger.addHandler(ch)
        if file_path is not None:
            fh = logging.FileHandler(file_path)
            fh.setFormatter(standard_formatter)
            self.logger.addHandler(fh)
    '''

    @abstractmethod
    def on_site_time(self, **kwargs) -> float:
        # I purposely left **kwargs here, because different models may require different parameters.
        # We can expand this class with a new method if we find out that there is a common set of parameters.
        raise NotImplementedError('Implement this method in your subclass')
