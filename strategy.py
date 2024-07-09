import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    <DESCRIPTION>
    User-made strategy to backtest.
    Output should be created as a dataframe with weights for each ticker and timeframe.
    1(Long), -1(Short), NaN(No action) will be valid option for its elements.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    @property
    def weights(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Weights dataframe containing tickers, timeframe, and its position(1, -1, NaN).
        """
        return


if __name__ == "__main__":
    pass
