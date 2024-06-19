import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from tools import *


class Performance:
    """
    <DESCRIPTION>
    Performance measure after backtesting investment strategies.
    Return data should be in daily basis.

    <PARAMS>
    pf_ret: Return of the portfolio.
    bm_ret: Return of the benchmark. If None, bm_ret will not be calculated.
    multiplier: Multiplier to measure performance with certain frequency.
    """

    class BMNotAvailableError(Exception):
        """
        <DESCRIPTION>
        Exception raised when BM return data is not available.
        """
        pass

    def __init__(self,
                 pf_ret: pd.DataFrame,
                 bm_ret: pd.DataFrame,
                 multiplier: str = 'D',
                 roll_multiplier: str = '6M'):
        multiplier_dict = {'Y': 252,
                           'M': 21,
                           'D': 1}
        self.multiplier_ = multiplier
        self.multiplier = Tools.validation_params_dict(value=multiplier,
                                                       valid_values=multiplier_dict)

        roll_multiplier_dict = {'Y': 12,
                                '9M': 9,
                                '6M': 6,
                                '3M': 3}
        self.roll_multiplier_ = roll_multiplier
        self.roll_multiplier = Tools.validation_params_dict(value=roll_multiplier,
                                                            valid_values=roll_multiplier_dict)

        self.pf_ret = pf_ret.astype(float)
        self.bm_ret = bm_ret.astype(float) if bm_ret is not None else None

        self.pf_dd = Tools.get_drawdown(cumret=self.pf_cumret)
        self.bm_dd = Tools.get_drawdown(
            cumret=self.bm_cumret) if self.bm_ret is not None else pd.DataFrame()

    def __call__(self):
        """
        Print every performance measures and plottings.
        """
        self.performance_plot().show()
        self.performance_plot_log_diff().show()
        self.performance_plot_rolling_ret().show()
        self.performance_plot_eoy().show()

        print(self.performance_table())
        print(f"\nTuW:\n{self.performance_tuw.head(5)}")
        print(f"\nEOY:\n{self.performance_eoy}")

    def __repr__(self):
        pass

    @property
    def pf_cumret(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Cumulative return of the portfolio.
        """
        return (1 + self.pf_ret.fillna(0)).cumprod()

    @property
    def bm_cumret(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Cumulative return of the benchmark.
        """
        return (1 + self.bm_ret.fillna(0)).cumprod()

    @property
    def pf_log_cumret(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Cumulative log return of the portfolio.
        """
        return np.log(1 + self.pf_ret.fillna(0)).cumsum()

    @property
    def bm_log_cumret(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Cumulative log return of the benchmark.
        """
        return np.log(1 + self.bm_ret.fillna(0)).cumsum()

    def performance_validation_check(self,
                                     pf_func: Callable[[pd.DataFrame], Any],
                                     bm_func: Callable[[pd.DataFrame], Any],
                                     default: Any = np.nan):
        """
        <DESCRIPTION>
        Check whether benchmark value exists, input NaN if not.

        <PARAMS>
        pf_func: Function to be used for portfolio return.
        bm_func: Function to be used for benchmark return.
        default: NaN value to input if benchmark value does not exists.
        """
        pf_value = pf_func(self.pf_ret)
        bm_value = bm_func(self.bm_ret) if self.bm_ret is not None else default
        return [pf_value, bm_value]

    @property
    def performance_mean(self) -> list:
        """
        <DESCRIPTION>
        Mean value of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: np.mean(pf, axis=0).iloc[0] * self.multiplier,
            lambda bm: np.mean(bm, axis=0).iloc[0] * self.multiplier
        )
        return res

    @property
    def performance_std(self) -> list:
        """
        <DESCRIPTION>
        Standard deviation value of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: np.std(pf, ddof=1, axis=0).iloc[
                0] * np.sqrt(self.multiplier),
            lambda bm: np.std(
                bm, ddof=1, axis=0).iloc[0] * np.sqrt(self.multiplier)
        )
        return res

    @property
    def performance_cagr(self) -> list:
        """
        <DESCRIPTION>
        CAGR value of portfolio cumulative return and benchmark cumulative return.
        """
        res = self.performance_validation_check(
            lambda pf: ((self.pf_cumret.iloc[-1] / self.pf_cumret.iloc[0])
                        ** (252 / len(self.pf_cumret)) - 1).iloc[0],
            lambda bm: ((self.bm_cumret.iloc[-1] / self.bm_cumret.iloc[0])
                        ** (252 / len(self.bm_cumret)) - 1).iloc[0]
        )
        return res

    @property
    def performance_sharpe_ratio(self) -> list:
        """
        <DESCRIPTION>
        Sharpe ratio of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: self.performance_mean[0] / self.performance_std[0],
            lambda bm: self.performance_mean[1] / self.performance_std[1]
        )
        return res

    @property
    def performance_sortino_ratio(self) -> list:
        def sortino_ratio(ret: pd.DataFrame):
            ret_downside = ret[ret < 0]
            std_downside = np.std(ret_downside,
                                  ddof=1,
                                  axis=0) * np.sqrt(self.multiplier)
            ret_res = ret.mean(axis=0).iloc[0] * self.multiplier
            std_res = std_downside.iloc[0]
            return (ret_res / std_res) if std_res != 0 else np.nan

        res = self.performance_validation_check(
            lambda pf: sortino_ratio(ret=self.pf_ret),
            lambda bm: sortino_ratio(ret=self.bm_ret)
        )
        return res

    @property
    def performance_calmar_ratio(self) -> list:
        res = self.performance_validation_check(
            lambda pf: self.performance_cagr[0] / abs(self.performance_mdd[0]),
            lambda bm: self.performance_cagr[1] / abs(self.performance_mdd[1])
        )
        return res

    @property
    def performance_skewness(self) -> list:
        """
        <DESCRIPTION>
        Shkewness of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: pf.skew().iloc[0],
            lambda bm: bm.skew().iloc[0]
        )
        return res

    @property
    def performance_kurtosis(self) -> list:
        """
        <DESCRIPTION>
        Kurtosis of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: pf.kurt().iloc[0],
            lambda bm: bm.kurt().iloc[0]
        )
        return res

    @property
    def performance_mdd(self) -> list:
        """
        <DESCRIPTION>
        MDD of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: np.min(self.pf_dd, axis=0).iloc[0],
            lambda bm: np.min(self.bm_dd, axis=0).iloc[0]
        )
        return res

    @property
    def performance_hit_ratio(self) -> list:
        """
        <DESCRIPTION>
        Hit ratio of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: len(pf[pf >= 0].dropna()) / len(pf),
            lambda bm: len(bm[bm >= 0].dropna()) / len(bm)
        )
        return res

    @property
    def performance_cumret(self) -> list:
        """
        <DESCRIPTION>
        Cumulative return value of portfolio return and benchmark return.
        """
        res = self.performance_validation_check(
            lambda pf: self.pf_cumret.iloc[-1].values[0] - 1,
            lambda bm: self.bm_cumret.iloc[-1].values[0] - 1
        )
        return res

    @property
    def performance_tuw(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Time under water of portfolio return.
        """
        df = pd.DataFrame()

        df['Underwater'] = self.pf_dd < 0
        df['Underwater_group'] = (
            df['Underwater'] != df['Underwater'].shift(periods=1)).cumsum()

        res = df[df['Underwater']].groupby('Underwater_group').agg(
            StartDate=('Underwater', lambda x: x.index.min()),
            EndDate=('Underwater', lambda x: x.index.max()),
            Duration=('Underwater', 'size')
        ).sort_values(by='Duration',
                      ascending=False).reset_index(drop=True)
        res['StartDate'] = res['StartDate'].dt.strftime('%Y-%m-%d')
        res['EndDate'] = res['EndDate'].dt.strftime('%Y-%m-%d')
        res.index.name = 'OrderBy'
        return res

    @property
    def performance_eoy(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        EOY return of portfolio and benchmark.
        """
        pf_ret = (1 + self.pf_ret.iloc[:, 0]
                  ).groupby(self.pf_ret.index.year).cumprod()
        pf_res = pf_ret.groupby(pf_ret.index.year).last()

        if self.bm_ret is not None:
            bm_ret = (
                1 + self.bm_ret.iloc[:, 0]).groupby(self.bm_ret.index.year).cumprod()
            bm_res = bm_ret.groupby(bm_ret.index.year).last()

            res = pd.concat([pf_res, bm_res], axis=1)
            res.columns = ['Portfolio', 'BM']
            res['ExcessRet'] = res['Portfolio'] - res['BM']
        else:
            res = pf_res.to_frame(name='Portfolio')
            res['BM'] = np.nan
            res['ExcessRet'] = np.nan

        res.index = res.index.astype(str)
        res.index.name = 'Year'
        res = np.round(res, 4)
        return res

    def performance_table(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Performance table of portfolio return and benchmark return.
        """
        def performance_pct_chg(perfs: list) -> list:
            """
            <DESCRIPTION>
            Change performance measures into pct, with rounding as 4.

            <PARAMS>
            perfs: Performance measures.
            """
            res = np.round([perf * 100 for perf in perfs], 4)
            return res

        invested = self.pf_ret.index

        msres = [performance_pct_chg(self.performance_mean),
                 performance_pct_chg(self.performance_std),
                 performance_pct_chg(self.performance_cagr),
                 np.round(self.performance_sharpe_ratio, 4),
                 np.round(self.performance_sortino_ratio, 4),
                 np.round(self.performance_calmar_ratio, 4),
                 np.round(self.performance_skewness, 4),
                 np.round(self.performance_kurtosis, 4),
                 performance_pct_chg(self.performance_mdd),
                 performance_pct_chg(self.performance_hit_ratio),
                 performance_pct_chg(self.performance_cumret)]

        res = pd.DataFrame(msres,
                           columns=[f"Performance ({self.multiplier_}, Portfolio)",
                                    f"Performance ({self.multiplier_}, BM)"],
                           index=['Mean (%)',
                                  'Standard Deviation (%)',
                                  'CAGR (%)',
                                  'Sharpe Ratio',
                                  'Sortino Ratio',
                                  'Calmar Ratio',
                                  'Skewness',
                                  'Kurtosis',
                                  'MDD (%)',
                                  'Hit Ratio (%)',
                                  'CumRet (%)'])

        res.index.name = f"{Tools.dt_to_str(invested[0])} ~ {Tools.dt_to_str(invested[-1])}"
        return res

    def performance_plot(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot cumulative return and MDD of portfolio and benchmark.
        """
        fig = go.Figure()
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(None, None))

        fig.add_trace(go.Scatter(x=self.pf_cumret.index,
                                 y=round(self.pf_cumret, 4).values.flatten(),
                                 mode='lines',
                                 name='Portfolio',
                                 line=dict(color='red')),
                      row=1,
                      col=1)
        if self.bm_ret is not None:
            fig.add_trace(go.Scatter(x=self.bm_cumret.index,
                                     y=round(self.bm_cumret,
                                             4).values.flatten(),
                                     mode='lines',
                                     name='Benchmark',
                                     line=dict(color='black')),
                          row=1,
                          col=1)
        fig.add_hline(y=1,
                      line=dict(color='rgba(0, 0, 0, 0.5)',
                                width=1,
                                dash='dash'),
                      row=1,
                      col=1)

        fig.add_trace(go.Scatter(x=self.pf_dd.index,
                                 y=round(self.pf_dd, 4).values.flatten(),
                                 mode='lines',
                                 name='Portfolio Drawdown',
                                 line=dict(color='red', dash='dot')),
                      row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=self.pf_dd.index,
                                 y=round(self.pf_dd, 4).values.flatten(),
                                 fill='tozeroy',
                                 mode='none',
                                 fillcolor='rgba(255, 0, 0, 0.2)',
                                 showlegend=False),
                      row=2,
                      col=1)
        if self.bm_ret is not None:
            fig.add_trace(go.Scatter(x=self.bm_dd.index,
                                     y=round(self.bm_dd, 4).values.flatten(),
                                     mode='lines',
                                     name='Benchmark Drawdown',
                                     line=dict(color='black', dash='dot')),
                          row=2,
                          col=1)
            fig.add_trace(go.Scatter(x=self.bm_dd.index,
                                     y=round(self.bm_dd, 4).values.flatten(),
                                     fill='tozeroy',
                                     mode='none',
                                     fillcolor='rgba(0, 0, 0, 0.2)',
                                     showlegend=False),
                          row=2,
                          col=1)

        fig.add_shape(type="line",
                      x0=0, x1=1, y0=0.5, y1=0.5,
                      xref="paper", yref="paper",
                      line=dict(color="black", width=0.5))

        y_min = min(self.pf_cumret.values.flatten().min(),
                    self.bm_cumret.values.flatten().min() if self.bm_ret is not None else float('inf'))
        y_max = max(self.pf_cumret.values.flatten().max(),
                    self.bm_cumret.values.flatten().max() if self.bm_ret is not None else float('-inf'))
        for _, row in self.performance_tuw[:3].iterrows():
            fig.add_shape(type="rect",
                          xref="x", yref="paper",
                          x0=row['StartDate'], y0=y_min,
                          x1=row['EndDate'], y1=y_max,
                          fillcolor="LightSalmon",
                          opacity=0.15,
                          layer="below",
                          line_width=0,
                          row=1,
                          col=1)

        fig.update_layout(
            **Tools.get_common_layout(title='Cumulative Return and Drawdown'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)')
        fig.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)')
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        return fig

    def performance_plot_log_diff(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot cumulative log return and difference of portfolio and benchmark.
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(None, None))

        fig.add_trace(go.Scatter(x=self.pf_log_cumret.index,
                                 y=round(self.pf_log_cumret,
                                         4).values.flatten(),
                                 mode='lines',
                                 name='Portfolio',
                                 line=dict(color='red')),
                      row=1,
                      col=1)
        if self.bm_ret is not None:
            fig.add_trace(go.Scatter(x=self.bm_log_cumret.index,
                                     y=round(self.bm_log_cumret,
                                             4).values.flatten(),
                                     mode='lines',
                                     name='Benchmark',
                                     line=dict(color='black')),
                          row=1,
                          col=1)

        fig.add_hline(y=0,
                      line=dict(color='rgba(0, 0, 0, 0.5)',
                                width=1,
                                dash='dash'),
                      row=1,
                      col=1)

        if self.bm_ret is not None:
            log_diff = pd.DataFrame(np.round(self.pf_log_cumret.values - self.bm_log_cumret.values, 4),
                                    index=self.pf_log_cumret.index)
            fig.add_trace(go.Scatter(x=log_diff.index,
                                     y=log_diff.values.flatten(),
                                     mode='lines',
                                     name='Difference',
                                     line=dict(color='grey')),
                          row=2,
                          col=1)

        fig.add_hline(y=0,
                      line=dict(color='rgba(0, 0, 0, 0.5)',
                                width=1,
                                dash='dash'),
                      row=2,
                      col=1)

        fig.add_shape(type="line",
                      x0=0, x1=1, y0=0.5, y1=0.5,
                      xref="paper", yref="paper",
                      line=dict(color="black", width=0.5))

        fig.update_layout(
            **Tools.get_common_layout(title='Log Cumulative Return and Difference'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)')
        fig.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)')
        fig.update_yaxes(title_text="Log Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Difference", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        return fig

    def performance_plot_rolling_ret(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot rolling return of portfolio and benchmark.
        """
        pf_rolling_ret_1m = Tools.get_rolling_ret(ret=self.pf_ret,
                                                  window=1,
                                                  multiplier=21,
                                                  roll_type='mean') * 100
        bm_rolling_ret_1m = Tools.get_rolling_ret(ret=self.bm_ret,
                                                  window=1,
                                                  multiplier=21,
                                                  roll_type='mean') * 100 if self.bm_ret is not None else None

        pf_rolling_ret_3m = Tools.get_rolling_ret(ret=self.pf_ret,
                                                  window=3,
                                                  multiplier=21,
                                                  roll_type='mean') * 100
        bm_rolling_ret_3m = Tools.get_rolling_ret(ret=self.bm_ret,
                                                  window=3,
                                                  multiplier=21,
                                                  roll_type='mean') * 100 if self.bm_ret is not None else None

        fig = make_subplots(rows=1,
                            cols=2,
                            subplot_titles=(None,
                                            None))
        fig.add_trace(go.Histogram(
            x=pf_rolling_ret_1m.values.flatten(),
            name='Portfolio 1M',
            marker_color='red',
            nbinsx=150,
            opacity=0.5,
            histnorm='probability density'),
            row=1,
            col=1)
        if bm_rolling_ret_1m is not None:
            fig.add_trace(go.Histogram(
                x=bm_rolling_ret_1m.values.flatten(),
                name='Benchmark 1M',
                marker_color='black',
                nbinsx=150,
                opacity=0.5,
                histnorm='probability density'),
                row=1,
                col=1)
        fig.add_vline(x=pf_rolling_ret_1m.mean().values[0],
                      line=dict(color='black', dash='dash'),
                      row=1,
                      col=1)

        fig.add_trace(go.Histogram(
            x=pf_rolling_ret_3m.values.flatten(),
            name='Portfolio 3M',
            marker_color='red',
            nbinsx=150,
            opacity=0.5,
            histnorm='probability density'),
            row=1,
            col=2)
        if bm_rolling_ret_3m is not None:
            fig.add_trace(go.Histogram(
                x=bm_rolling_ret_3m.values.flatten(),
                name='Benchmark 3M',
                marker_color='black',
                nbinsx=150,
                opacity=0.5,
                histnorm='probability density'),
                row=1,
                col=2)
        fig.add_vline(x=pf_rolling_ret_3m.mean().values[0],
                      line=dict(color='black', dash='dash'),
                      row=1,
                      col=2)

        fig.update_layout(
            **Tools.get_common_layout(title='Rolling Return Histograms'),
            barmode='overlay'
        )
        fig.update_yaxes(title_text='Probability Density',
                         tickformat='.0%', row=1, col=1)
        fig.update_yaxes(title_text='Probability Density',
                         tickformat='.0%', row=1, col=2)
        fig.update_xaxes(title_text='1-Month Rolling Return (%)', row=1, col=1)
        fig.update_xaxes(title_text='3-Month Rolling Return (%)', row=1, col=2)
        return fig

    def performance_plot_rolling_sharpe(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot rolling sharpe ratio of portfolio and benchmark.
        """
        pf_rolling_mean = Tools.get_rolling_ret(ret=self.pf_ret,
                                                window=self.roll_multiplier,
                                                multiplier=21,
                                                roll_type='mean')
        pf_rolling_std = Tools.get_rolling_ret(ret=self.pf_ret,
                                               window=self.roll_multiplier,
                                               multiplier=21,
                                               roll_type='std')

        bm_rolling_mean = Tools.get_rolling_ret(ret=self.bm_ret,
                                                window=self.roll_multiplier,
                                                multiplier=21,
                                                roll_type='mean') if self.bm_ret is not None else None
        bm_rolling_std = Tools.get_rolling_ret(ret=self.bm_ret,
                                               window=self.roll_multiplier,
                                               multiplier=21,
                                               roll_type='std') if self.bm_ret is not None else None

        pf_rolling_sharpe = round(
            (pf_rolling_mean / pf_rolling_std) * self.multiplier, 4)
        bm_rolling_sharpe = round(
            (bm_rolling_mean / bm_rolling_std) * self.multiplier, 4)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pf_rolling_sharpe.index,
            y=pf_rolling_sharpe.values.flatten(),
            mode='lines',
            name='Portfolio',
            line=dict(color='red')
        ))

        if self.bm_ret is not None:
            fig.add_trace(go.Scatter(x=bm_rolling_sharpe.index,
                                     y=bm_rolling_sharpe.values.flatten(),
                                     mode='lines',
                                     name='Benchmark',
                                     line=dict(color='black')))

        fig.add_hline(y=0,
                      line=dict(color='rgba(0, 0, 0, 0.5)',
                                width=1,
                                dash='dash'))

        fig.update_layout(
            **Tools.get_common_layout(title=f'Rolling Sharpe Ratio (Window {self.roll_multiplier_}, Frequency {self.multiplier_})'),
            yaxis_title='Sharpe Ratio',
            xaxis_title='Date'
        )
        return fig

    def performance_plot_eoy(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot EOY return of portfolio and benchmark.
        """
        df = self.performance_eoy.copy()
        df['Portfolio'] = (df['Portfolio'] - 1) * 100
        if 'BM' in df.columns and df['BM'].notnull().all():
            df['BM'] = (df['BM'] - 1) * 100
        else:
            pass

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Portfolio'],
            name='Portfolio',
            marker_color='red'
        ))

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['BM'],
            name='Benchmark',
            marker_color='black'
        ))

        fig.update_layout(**Tools.get_common_layout(title='EOY Portfolio and Benchmark Cumulative Return'),
                          yaxis_title='Return (%)',
                          xaxis_title='Year',
                          barmode='group')

        fig.add_shape(type='line',
                      x0=min(df.index), y0=0,
                      x1=max(df.index), y1=0,
                      line=dict(color='black', width=1))
        return fig

    def performance_plot_ret_specific(self) -> go.Figure:
        """
        <DESCRIPTION>
        Plot return boxplot, heatmap of portfolio and benchmark.
        """
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(None, None))

        daily_ret = self.pf_ret
        weekly_ret = self.pf_ret.resample(
            'W').apply(lambda x: (x + 1).prod() - 1)
        monthly_ret = self.pf_ret.resample(
            'M').apply(lambda x: (x + 1).prod() - 1)
        quarterly_ret = self.pf_ret.resample(
            'Q').apply(lambda x: (x + 1).prod() - 1)
        yearly_ret = self.pf_ret.resample(
            'Y').apply(lambda x: (x + 1).prod() - 1)

        colors = [
            'rgba(31, 119, 180, 0.2)',
            'rgba(31, 119, 180, 0.4)',
            'rgba(31, 119, 180, 0.6)',
            'rgba(31, 119, 180, 0.8)',
            'rgba(31, 119, 180, 1.0)'
        ]

        fig.add_trace(go.Box(
            y=daily_ret.squeeze() * 100,
            name='Daily',
            boxmean='sd',
            marker_color=colors[0],
            hovertemplate='%{y:.4f}%'
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=weekly_ret.squeeze() * 100,
            name='Weekly',
            boxmean='sd',
            marker_color=colors[1],
            hovertemplate='%{y:.4f}%'
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=monthly_ret.squeeze() * 100,
            name='Monthly',
            boxmean='sd',
            marker_color=colors[2],
            hovertemplate='%{y:.4f}%'
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=quarterly_ret.squeeze() * 100,
            name='Quarterly',
            boxmean='sd',
            marker_color=colors[3],
            hovertemplate='%{y:.4f}%'
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=yearly_ret.squeeze() * 100,
            name='Yearly',
            boxmean='sd',
            marker_color=colors[4],
            hovertemplate='%{y:.4f}%'
        ), row=1, col=1)

        pf_ret_monthly = monthly_ret.copy()
        pf_ret_monthly['Year'] = pf_ret_monthly.index.year
        pf_ret_monthly['Month'] = pf_ret_monthly.index.strftime('%b')
        pf_ret_pivot = pf_ret_monthly.pivot_table(
            values='Portfolio', index='Year', columns='Month', aggfunc='sum')
        pf_ret_pivot = pf_ret_pivot.sort_index(ascending=False)
        months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                        'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        pf_ret_pivot = pf_ret_pivot[months_order]
        pf_ret_pivot = pf_ret_pivot * 100
        pf_ret_pivot = np.round(pf_ret_pivot, 2)

        fig.add_trace(go.Heatmap(
            z=pf_ret_pivot.values,
            x=pf_ret_pivot.columns,
            y=pf_ret_pivot.index,
            colorscale='RdYlGn',
            zmin=-30,
            zmax=30,
            text=pf_ret_pivot.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            showscale=False
        ), row=1, col=2)

        fig.update_layout(
            **Tools.get_common_layout(title="Return Analysis (%)"),
            yaxis_title='Return (%)',
            xaxis_title='Frequency',
            showlegend=False
        )

        fig.update_xaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Year", row=1, col=2, autorange="reversed")
        return fig


if __name__ == "__main__":
    ret = pd.read_pickle('./ret.pkl')
    bm = pd.read_pickle('./bm.pkl')

    perf = Performance(pf_ret=ret,
                       bm_ret=bm,
                       multiplier='D',
                       roll_multiplier='6M')
