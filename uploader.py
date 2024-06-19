import streamlit as st

from performance import *


class StreamlitUploader(Performance):
    """
    <DESCRIPTION>
    Upload performance measures to web using streamlit.
    """

    def __init__(self,
                 pf_ret: pd.DataFrame,
                 bm_ret: pd.DataFrame,
                 multiplier: str = 'Y',
                 roll_multiplier: str = '6M'):
        super().__init__(pf_ret, bm_ret, multiplier, roll_multiplier)
        self.pf_ret = pf_ret
        self.bm_ret = bm_ret

    def upload(self):
        """
        <DESCRIPTION>
        Upload streamlit.
        """
        st.divider()

        st.markdown("<h1 id='performance-analysis' style='text-align: center; color: black; font-size: 24px;'>Performance Analysis</h1>",
                    unsafe_allow_html=True)

        with st.expander("Cumulative Return and Drawdown", expanded=True):
            st.plotly_chart(self.performance_plot(), use_container_width=True)

        with st.expander("Log Cumulative Return and Difference", expanded=True):
            st.plotly_chart(self.performance_plot_log_diff(),
                            use_container_width=True)

        with st.expander("Rolling Return Histograms", expanded=True):
            st.plotly_chart(self.performance_plot_rolling_ret(),
                            use_container_width=True)

        with st.expander("EOY Portfolio and Benchmark Cumulative Return", expanded=True):
            st.plotly_chart(self.performance_plot_eoy(),
                            use_container_width=True)

        with st.expander("Rolling Sharpe Ratio", expanded=True):
            st.plotly_chart(self.performance_plot_rolling_sharpe(),
                            use_container_width=True)

        with st.expander("Return Analysis", expanded=True):
            st.plotly_chart(self.performance_plot_ret_specific(),
                            use_container_width=True)

        st.divider()

        st.markdown("<h1 id='performance-metrics' style='text-align: center; color: black; font-size: 24px;'>Performance Metrics</h1>",
                    unsafe_allow_html=True)

        with st.expander("Performance Metrics Tables", expanded=True):
            tabs = st.tabs(
                ["Performance Table", "Time Under Water", "EOY Performance"])
            with tabs[0]:
                st.caption('Performance Table')
                st.dataframe(self.performance_table(),
                             use_container_width=True)

            with tabs[1]:
                st.caption('Time Under Water')
                st.dataframe(self.performance_tuw, use_container_width=True)

            with tabs[2]:
                st.caption('EOY Performance')
                performance_eoy = self.performance_eoy.copy()
                performance_eoy['Portfolio'] = performance_eoy['Portfolio'] - 1
                performance_eoy['BM'] = performance_eoy['BM'] - 1
                performance_eoy = performance_eoy * 100
                st.dataframe(performance_eoy.style.format(
                    "{:.2f}%"), use_container_width=True)
