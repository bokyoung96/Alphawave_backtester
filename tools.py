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

    @staticmethod
    def get_rolling_ret(ret: pd.DataFrame,
                        window: int,
                        multiplier: int,
                        roll_type: str):
        """
        <DESCRIPTION>
        Get average rolling return of the portfolio.

        <PARAMS>
        ret: Return of the portfolio.
        window: Size of the rolling window.
        multiplier: Multiplier of the portfolio considering frequency.
        """
        if roll_type == 'mean':
            rolling_ret = ret.rolling(
                window * multiplier).mean().dropna(axis=0)
        elif roll_type == 'std':
            rolling_ret = ret.rolling(window * multiplier).std().dropna(axis=0)
        return rolling_ret

    @staticmethod
    def get_common_layout(title: str) -> dict:
        """
        <DESCRIPTION>
        Get common layout for plots.

        <PARAMS>
        title: Title of the plot.
        """
        return dict(title=title,
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=600,
                    width=1000,
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True,
                               gridcolor='rgba(200, 200, 200, 0.5)'),
                    yaxis=dict(showgrid=True,
                               gridcolor='rgba(200, 200, 200, 0.5)'),
                    legend=dict(orientation='h',
                                yanchor='top',
                                y=-0.2,
                                xanchor='right',
                                x=1))
