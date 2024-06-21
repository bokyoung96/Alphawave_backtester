import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import to_hex
from plotly.subplots import make_subplots

from tools import *


class Contribution:
    """
    <DESCRIPTION>
    Contribution measure after backtesting investment strategies.
    Weight data should be in rebalancing period basis.

    <PARAMS>
    weight: PDF weights for each rebalancing periods.
    multiplier: Multiplier to measure performance with certain frequency.
    """

    def __init__(self,
                 weight: pd.DataFrame,
                 contribution_multiplier: str,
                 start_date: str,
                 end_date: str):
        multiplier_dict = {'Y': 252,
                           'M': 21,
                           'D': 1}
        self.contribution_multiplier_ = contribution_multiplier
        self.multiplier = Tools.validation_params_dict(value=contribution_multiplier,
                                                       valid_values=multiplier_dict)

        self.weight = weight.astype(
            float) if weight is not None else pd.DataFrame()

        self.start_date = start_date
        self.end_date = end_date

        # NOTE: Should be refactored afterwards.
        # NOTE: Contact BKChoi for datas - LFS required.
        self.price = pd.read_pickle(
            './loader_data/KOSPI_stock_price_c_1d_quantiwise.pkl')
        self.names = pd.read_pickle('./loader_data/KOSPI_names_quantiwise.pkl')
        self.sectors = pd.read_pickle(
            './loader_data/KOSPI_sectors_quantiwise.pkl')

    @property
    def w(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Preprocess weight of the portfolio.
        """
        df = self.weight.reset_index(drop=False).rename(
            columns={'index': 'Date'})
        res = df.melt(id_vars=['Date'],
                      var_name='Code',
                      value_name='Weight').dropna(axis=0,
                                                  how='any')
        res = res.sort_values(by='Date').set_index('Date')
        return res

    def contribution_w(self,
                       top: bool = True) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Calculate the contribution of each PDF.

        <PARAMS>
        top: Select whether to show top or the bottom.
        """
        w = self.w.loc[self.start_date: self.end_date]
        code = w['Code'].unique()

        price = self.price[code].loc[self.start_date: self.end_date]
        ret = price.pct_change(fill_method=None,
                               axis=0)
        cumret = (1 + ret).cumprod(axis=0)

        res_ret = ret.mean(axis=0)
        res_cumret = cumret.iloc[-1, :]

        res = pd.concat([res_ret, res_cumret], axis=1)
        res.columns = [f"MeanRet ({self.contribution_multiplier_})",
                       "CumRet"]
        res[f"MeanRet ({self.contribution_multiplier_})"] = res[f"MeanRet ({self.contribution_multiplier_})"] * self.multiplier
        res.index.name = 'Ticker'

        if top:
            res = res.head(10)
        else:
            res = res.tail(10)

        for df in [self.sectors, self.names]:
            res = pd.merge(df, res, how='inner',
                           left_index=True,
                           right_index=True).sort_values(by="CumRet",
                                                         ascending=False)
        return res

    def contribution_plot_w_ticker(self,
                                   top: bool = True) -> go.Figure:
        """
        <DESCRIPTION>
        Plot top contributors of the PDF.

        <PARAMS>
        top: Select whether to show top or the bottom.
        """
        tickers = self.contribution_w(top=top).index
        price = self.price[tickers].loc[self.start_date: self.end_date]
        ret = price.pct_change(fill_method=None, axis=0)
        cumret = (1 + ret.fillna(0)).cumprod(axis=0)

        sorted_tickers = cumret.iloc[-1].sort_values(ascending=False).index

        cmap = plt.colormaps['coolwarm']
        colors = [to_hex(cmap(i / (len(sorted_tickers) - 1)))
                  for i in range(len(sorted_tickers))]

        fig = go.Figure()

        for idx, ticker in enumerate(sorted_tickers):
            fig.add_trace(go.Scatter(
                x=cumret.index,
                y=cumret[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=1, color=colors[-(idx + 1)])
            ))

        if top:
            fig.update_layout(**Tools.get_common_layout(title='Top Return Contributors'),
                              yaxis_title='Cumulative Return',
                              xaxis_title='Year')
        else:
            fig.update_layout(**Tools.get_common_layout(title='Bottom Return Contributors'),
                              yaxis_title='Cumulative Return',
                              xaxis_title='Year')

        return fig

    def contribution_plot_w_sector(self,
                                   top: bool = True) -> go.Figure:
        """
        <DESCRIPTION>
        Plot top contributor sectors of the PDF.

        <PARAMS>
        top: Select whether to show top or the bottom.
        """
        sectors = self.contribution_w(
            top=top)['Sector'].value_counts().reset_index()
        sectors.columns = ['Sector', 'count']

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=sectors['Sector'],
            values=sectors['count'],
            hole=0.4,
            pull=[0.1] * len(sectors),
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2))
        ))

        if top:
            fig.update_layout(
                **Tools.get_common_layout(title='Top Sector Distribution'))
        else:
            fig.update_layout(
                **Tools.get_common_layout(title='Bottom Sector Distribution'))
        return fig


if __name__ == "__main__":
    weight = pd.read_pickle('./sample_data/weight.pkl')

    contr = Contribution(weight=weight,
                         contribution_multiplier='D',
                         start_date='20200101',
                         end_date='20201231')
