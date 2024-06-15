import numpy as np
import pandas as pd
import datetime as dt


from typing import *


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def dt_to_str(idx: pd.Timestamp) -> str:
        return dt.datetime.strftime(idx, '%Y%m%d')

    @staticmethod
    def validation_params_dict(value: Any,
                               valid_values: dict):
        """
        <DESCRIPTION>
        Validate whether the value selected is included in dictionary.

        <PARAMS>
        value: Value to validate.
        valid_values: Dictionary to classify whether the value is included in.
        """
        if isinstance(valid_values, dict):
            try:
                return valid_values[value]
            except KeyError:
                raise ValueError(
                    f"Invalid option: {value}.\nAllowed values are: {', '.join(map(str, valid_values.keys()))}.")
        else:
            raise TypeError(
                "Parameter <valid_values> should be dictionary type.")

    @staticmethod
    def get_drawdown(cumret: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Get drawdown of the portfolio.

        <PARAMS>
        cumret: Cumulative return of the portfolio.
        """
        peak = cumret.cummax()
        drawdown = (cumret - peak) / peak
        drawdown.replace([np.inf, -np.inf], np.nan, inplace=True)
        drawdown.fillna(0, inplace=True)
        return drawdown
