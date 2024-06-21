import streamlit as st

from performance import *
from contribution import *


class StreamlitUploader(Performance, Contribution):
    """
    <DESCRIPTION>
    Upload performance measures to web using streamlit.
    """

    def __init__(self,
                 pf_ret: pd.DataFrame,
                 bm_ret: pd.DataFrame,
                 performance_multiplier: str,
                 roll_multiplier: str,
                 weight: pd.DataFrame,
                 contribution_multiplier: str,
                 start_date: str,
                 end_date: str):
        Performance.__init__(self, pf_ret, bm_ret,
                             performance_multiplier, roll_multiplier)
        Contribution.__init__(
            self, weight, contribution_multiplier, start_date, end_date)
        self.pf_ret = pf_ret
        self.bm_ret = bm_ret
        self.weight = weight

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

        st.divider()
        st.markdown("<h1 id='contribution-analysis' style='text-align: center; color: black; font-size: 24px;'>Contribution Analysis</h1>",
                    unsafe_allow_html=True)

        if self.weight is not None:
            with st.expander("Contribution Metrics Tables", expanded=True):
                st.markdown("""
                            <div style="background-color: #fff3cd; padding: 10px; border-left: 5px solid #ffeeba;">
                                <strong>Note:</strong> This table shows the return and sector classification of the stocks that had the highest returns 
                                within the set start and end dates, and were held at least once during this period.<br>
                                The actual contribution analysis based on the holding period is still under development.
                            </div>
                            """, unsafe_allow_html=True)

                tabs_w = st.tabs(["Return Contributors",
                                  "Return Contributors Plot",
                                  "Sector Contributions"])
                with tabs_w[0]:
                    st.caption('Top Return Contributors')
                    contr_w = self.contribution_w(top=True).copy()
                    contr_w = contr_w.style.format({
                        f"MeanRet ({self.contribution_multiplier_})": Tools.format_pct,
                        "CumRet": Tools.format_pct
                    })
                    st.dataframe(contr_w, use_container_width=True)

                    st.caption('Bottom Return Contributors')
                    contr_w = self.contribution_w(top=False).copy()
                    contr_w = contr_w.style.format({
                        f"MeanRet ({self.contribution_multiplier_})": Tools.format_pct,
                        "CumRet": Tools.format_pct
                    })
                    st.dataframe(contr_w, use_container_width=True)

                with tabs_w[1]:
                    st.caption('Top Return Contributors Plot')
                    st.plotly_chart(self.contribution_plot_w_ticker(
                        top=True), use_container_width=True)

                    st.caption('Bottom Return Contributors Plot')
                    st.plotly_chart(self.contribution_plot_w_ticker(
                        top=False), use_container_width=True)

                with tabs_w[2]:
                    st.caption('Top Sector Contribution Plot')
                    st.plotly_chart(self.contribution_plot_w_sector(
                        top=True), use_container_width=True)

                    st.caption('Bottom Sector Contribution Plot')
                    st.plotly_chart(self.contribution_plot_w_sector(
                        top=False), use_container_width=True)

        else:
            st.warning(
                "Contribution analysis is skipped since weight data was not uploaded. Please upload weight data to view the contribution analysis.")
