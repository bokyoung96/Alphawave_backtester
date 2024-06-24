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

    @property
    def w_reidx(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Reindex the weight of the portfolio to daily basis.

        <NOTE>
        1. Date class for numerous countries / asset classes will be required afterwards.
        2. Considering backtesting tool to be developed, [w_reidx] is temp code to use.
        """
        df = self.weight[self.start_date: self.end_date]
        date_range = self.price[self.start_date: self.end_date].index

        nan_mask = df.isna()
        w_reidx = df.reindex(date_range).ffill()

        nan_mask_reidx = nan_mask.reindex(
            date_range, method='ffill').fillna(False)
        w_reidx = w_reidx.where(~nan_mask_reidx, np.nan)
        return w_reidx

    def contribution_w_holdings(self, top: bool = True) -> pd.DataFrame:
        """
        Calculate the contribution of each PDF which user had at least once holded.

        Params:
        top: Select whether to show top or the bottom.
        """
        ret = self.price.loc[self.start_date:self.end_date].pct_change(
            fill_method=None)
        w_reidx = self.w_reidx

        temp_w = ret * w_reidx
        temp_w_ = ret * Tools.df_chgr(w_reidx)

        temp_sum = temp_w.sum(axis=0)
        temp_sum_ = temp_w_.sum(axis=0)

        temp_cumret = (temp_w.fillna(0) + 1).cumprod().iloc[-1] - 1
        temp_cumret_ = (temp_w_.fillna(0) + 1).cumprod().iloc[-1] - 1

        ret_mean = ret.mean(axis=0) * self.multiplier
        ret_cumul = (ret.fillna(0) + 1).cumprod().iloc[-1] - 1

        res = pd.DataFrame({
            f'Ret (HP / WA / SUM)': temp_sum,
            f'Ret (HP / WU / SUM)': temp_sum_,
            'Ret (HP / WA / CUMUL)': temp_cumret,
            'Ret (HP / WU / CUMUL)': temp_cumret_,
            f'Ret (WP / WU / MEAN / {self.contribution_multiplier_})': ret_mean,
            'Ret (WP / WU / CUMUL)': ret_cumul
        }).sort_values(by=f'Ret (HP / WA / SUM)', ascending=False)

        res = res.head(10) if top else res.tail(10)

        res = res.join(self.sectors, how='left').join(self.names, how='left')
        res.index.name = 'Ticker'

        col_order = ['Name', 'Sector'] + \
            [col for col in res.columns if col not in ['Name', 'Sector']]
        res = res[col_order]
        return res

    def contribution_plot_w_ret(self,
                                top: bool = True) -> go.Figure:
        """
        <DESCRIPTION>
        Plot top contributors of the PDF considering holding periods.

        <PARAMS>
        top: Select whether to show top or the bottom.
        """
        ret_temp = self.price.loc[self.start_date: self.end_date].pct_change(
            fill_method=None)
        ret = ret_temp * Tools.df_chgr(self.w_reidx)
        tickers = self.contribution_w_holdings(top=top).index

        cumret = round((1 + ret[tickers].fillna(0)).cumprod() - 1, 4)
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

    def contribution_plot_w_ticker(self,
                                   top: bool = True) -> go.Figure:
        """
        <DESCRIPTION>
        Plot top contributors of the PDF.

        <PARAMS>
        top: Select whether to show top or the bottom.

        <NOTE>
        Not used temporarily.
        """
        tickers = self.contribution_w_holdings(top=top).index
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

    def contribution_plot_w_sector(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot top / bottom contributor sectors of the PDF.
        """
        sectors_top = self.contribution_w_holdings(
            top=True)['Sector'].value_counts().reset_index()
        sectors_bottom = self.contribution_w_holdings(
            top=False)['Sector'].value_counts().reset_index()

        sectors_top.columns = ['Sector', 'count']
        sectors_bottom.columns = ['Sector', 'count']

        fig = make_subplots(
            rows=1, cols=2, shared_xaxes=True,
            horizontal_spacing=0.02,
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=("Top Sectors", "Bottom Sectors"))

        fig.add_trace(go.Pie(
            labels=sectors_top['Sector'],
            values=sectors_top['count'],
            hole=0.4,
            pull=[0.1] * len(sectors_top),
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2)),
        ),
            row=1, col=1)

        fig.add_trace(go.Pie(
            labels=sectors_bottom['Sector'],
            values=sectors_bottom['count'],
            hole=0.4,
            pull=[0.1] * len(sectors_bottom),
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2)),
        ),
            row=1, col=2)

        fig.update_layout(
            **Tools.get_common_layout(title='Top / Bottom Sector Contribution'))
        return fig

    def contribution_plot_w_sector_all(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot whole period contributor sectors of the PDF.
        """
        w = self.weight[self.start_date: self.end_date].sum(
            axis=0).sort_values(ascending=False)
        w = w / w.sum()

        w_rescaled = w[w != 0]
        w_rescaled = pd.DataFrame(w_rescaled, columns=['Weights'])

        sectors = self.sectors.fillna('ETC')

        temp = pd.merge(sectors, w_rescaled,
                        how='inner',
                        left_index=True,
                        right_index=True)
        res = temp.groupby('Sector')['Weights'].sum().reset_index()

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=res['Sector'],
            values=res['Weights'],
            hole=0.4,
            pull=[0.1] * len(res),
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2))
        ))

        fig.update_layout(
            **Tools.get_common_layout(title='Whole Period Sector Contribution'))
        return fig

    def contribution_plot_w_nums(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot the number of stocks invested by timeframe.
        """
        nums = pd.DataFrame(self.weight.notna().astype(int).sum(axis=1),
                            columns=['Nums'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nums.index,
            y=nums['Nums'],
            mode='lines+markers',
            line=dict(color='black'),
            marker=dict(color='grey'),
            name='Number invested'
        ))

        fig.add_hline(y=nums['Nums'].mean(),
                      line=dict(color='red',
                                width=2,
                                dash='dash'))

        fig.update_layout(**Tools.get_common_layout(title='Number of Items Invested'),
                          xaxis_title='Date',
                          yaxis_title='Number invested',
                          xaxis_tickformat='%Y-%m')
        return fig

    def contribution_turnover(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Calculate turnover rate.
        """
        w_diff = self.w_reidx.diff().abs().sum(axis=1)
        to_y = w_diff.mean() * 252
        to_6m = w_diff.mean() * 21 * 6
        to_m = w_diff.mean() * 21
        to_d = w_diff.mean()

        res = pd.DataFrame([to_y, to_6m, to_m, to_d],
                           index=['Y', '6M', 'M', 'D'],
                           columns=['Turnover Rate']).T
        return res


if __name__ == "__main__":
    weight = pd.read_pickle('./sample_data/weight.pkl')

    contr = Contribution(weight=weight,
                         contribution_multiplier='D',
                         start_date='20200101',
                         end_date='20201231')
