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
                        roll_type: str,
                        bm_ret: pd.DataFrame = None):
        """
        <DESCRIPTION>
        Get average rolling return of the portfolio.

        <PARAMS>
        ret: Return of the portfolio.
        window: Size of the rolling window.
        multiplier: Multiplier of the portfolio considering frequency.
        df: Benchmark return to use when calculating rolling correlation.
        """
        if roll_type == 'mean':
            rolling_ret = ret.rolling(
                window * multiplier).mean().dropna(axis=0)
        elif roll_type == 'std':
            rolling_ret = ret.rolling(window * multiplier).std().dropna(axis=0)
        elif roll_type == 'corr':
            if bm_ret is None:
                return None
            rolling_ret = ret.dropna().iloc[:, 0].rolling(
                window * multiplier).corr(bm_ret.dropna().iloc[:, 0]).dropna(axis=0)
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

    @staticmethod
    def format_pct(value: format) -> str:
        """
        <DESCRIPTION>
        Get percentage format for values.

        <PARAMS>
        value: The value to be formatted as a percentage.
        """
        return f"{value * 100:.2f}%"

    @staticmethod
    def df_chgr(df: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Change the value of numbers in dataframe into 1, leaving NaN as NaN.

        <PARAMS>
        df: Dataframe to be cahnged.
        """
        df_vals = df.values
        df_vals[~np.isnan(df_vals)] = 1
        res = pd.DataFrame(df_vals,
                           index=df.index,
                           columns=df.columns)
        return res
