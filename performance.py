import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools import *


class Performance:
    """
    <DESCRIPTION>
    Performance measure after backtesting investment strategies.

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
                 multiplier: str = 'D'):
        multiplier_dict = {'Y': 252,
                           'M': 20,
                           'D': 1}
        self.multiplier_ = multiplier
        self.multiplier = Tools.validation_params_dict(value=multiplier,
                                                       valid_values=multiplier_dict)

        self.pf_ret = pf_ret.astype(float)
        self.bm_ret = bm_ret.astype(float) if bm_ret is not None else None

        self.pf_dd = Tools.get_drawdown(cumret=self.pf_cumret)
        self.bm_dd = Tools.get_drawdown(
            cumret=self.bm_cumret) if self.bm_ret is not None else pd.DataFrame()

    def __repr__(self):
        pass

    def __call__(self):
        """
        <DESCRIPTION>
        Print every performance measures and plottings.
        """
        self.performance_plot()
        self.performance_plot_log_diff()

        print(self.performance_table())
        print(f"\nTuW:\n{self.performance_tuw.head(5)}")

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
    def performance_calmar_ratio(self) -> list:
        res = self.performance_validation_check(
            lambda pf: self.performance_cagr[0] / abs(self.performance_mdd[0]),
            lambda bm: self.performance_cagr[1] / abs(self.performance_mdd[1])
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
            Start_date=('Underwater', lambda x: x.index.min()),
            End_date=('Underwater', lambda x: x.index.max()),
            Duration=('Underwater', 'size')
        ).sort_values(by='Duration',
                      ascending=False).reset_index(drop=True)
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
                 np.round(self.performance_calmar_ratio, 4),
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
                                  'Calmar Ratio',
                                  'MDD (%)',
                                  'Hit Ratio (%)',
                                  'CumRet (%)'])

        res.index.name = f"{Tools.dt_to_str(invested[0])} ~ {Tools.dt_to_str(invested[-1])}"
        return res

    def performance_plot(self) -> plt.plot:
        """
        <DESCRIPTION>
        Plot cumulative return and MDD of portfolio and benchmark.
        """
        fig, axs = plt.subplots(2, 1,
                                sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

        axs[0].plot(self.pf_cumret,
                    label='Portfolio',
                    color='red',
                    linewidth=2,
                    linestyle='-')
        if self.bm_ret is not None:
            axs[0].plot(self.bm_cumret,
                        label='BM',
                        color='black',
                        linewidth=1.5,
                        linestyle='--')
        axs[0].set_ylabel('Cumulative return')
        axs[0].set_title('Portfolio versus BM cumulative return')
        axs[0].legend(loc='best')

        axs[1].plot(self.pf_dd,
                    label='Portfolio drawdown',
                    color='red',
                    linewidth=2,
                    linestyle='-')
        axs[1].fill_between(self.pf_dd.index.to_pydatetime(),
                            self.pf_dd.values.flatten(),
                            color='red',
                            alpha=0.1)
        if self.bm_ret is not None:
            axs[1].plot(self.bm_dd,
                        label='BM drawdown',
                        color='black',
                        linewidth=1.5,
                        linestyle='--')
            axs[1].fill_between(self.bm_dd.index.to_pydatetime(),
                                self.bm_dd.values.flatten(),
                                color='black',
                                alpha=0.1)
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Drawdown')

        plt.tight_layout()
        plt.show()

    def performance_plot_log_diff(self) -> plt.plot:
        """
        <DESCRIPTION>
        Plot cumulative log return and MDD of the difference between portfolio and benchmark.
        """
        fig, axs = plt.subplots(2, 1,
                                sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
        if self.bm_ret is not None:
            log_diff = pd.DataFrame(self.pf_log_cumret.values - self.bm_log_cumret.values,
                                    index=self.pf_log_cumret.index)
            log_diff_dd = Tools.get_drawdown(cumret=log_diff)

            axs[0].plot(log_diff,
                        label='Log difference between Portfolio and BM',
                        color='blue',
                        linewidth=1.5,
                        linestyle='-')
            axs[0].set_ylabel('Cumulative log return difference')
            axs[0].set_title(
                'Portfolio versus BM cumulative log return difference')
            axs[0].legend(loc='best')

            axs[1].plot(log_diff_dd,
                        label='Log difference drawdown',
                        color='blue',
                        linewidth=1.5,
                        linestyle='-')
            axs[1].fill_between(log_diff_dd.index.to_pydatetime(),
                                log_diff_dd.values.flatten(),
                                color='blue',
                                alpha=0.1)
            axs[1].set_ylabel('Drawdown')
            axs[1].set_xlabel('Date')

            plt.tight_layout()
            plt.show()

        else:
            raise self.BMNotAvailableError(
                "Benchmark return data is not available.")


if __name__ == "__main__":
    ret = pd.read_pickle('./ret.pkl')
    # bm = pd.read_pickle('./bm.pkl')

    perf = Performance(pf_ret=ret,
                       bm_ret=ret,
                       multiplier='D')
