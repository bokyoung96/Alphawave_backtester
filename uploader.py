import streamlit as st

from performance import *

"""
# NOTE
pip install -r requirements.txt
cmd - location - streamlit run uploader.py
"""

class StreamlitUploader(Performance):
    """
    <DESCRIPTION>
    Upload performance measures to web using streamlit.
    """
    def __init__(self, 
                 pf_ret: pd.DataFrame, 
                 bm_ret: pd.DataFrame, 
                 multiplier: str = 'D'):
        super().__init__(pf_ret, bm_ret, multiplier)
        self.pf_ret = pf_ret
        self.bm_ret = bm_ret
        
    def run(self):
        """
        <DESCRIPTION>
        Run streamlit.
        """
        st.set_page_config(layout='wide')

        st.title('Performance Analysis')

        st.subheader('Cumulative Return and Drawdown')
        st.plotly_chart(self.performance_plot(), use_container_width=True)

        st.subheader('Log Cumulative Return and Difference')
        st.plotly_chart(self.performance_plot_log_diff(), use_container_width=True)

        st.subheader('Rolling Return Histograms')
        st.plotly_chart(self.performance_plot_rolling_ret(), use_container_width=True)

        st.subheader('EOY Portfolio and Benchmark Cumulative Return')
        st.plotly_chart(self.performance_plot_eoy(), use_container_width=True)

        st.subheader('Performance Metrics')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption('Performance Table')
            st.dataframe(self.performance_table())

        with col2:
            st.caption('Time Under Water')
            st.dataframe(self.performance_tuw)

        with col3:
            st.caption('EOY Performance')
            st.dataframe(self.performance_eoy)
            

if __name__ == "__main__":
    ret = pd.read_pickle('./ret.pkl')
    bm = pd.read_pickle('./bm.pkl')
    
    uploader = StreamlitUploader(pf_ret=ret,
                                 bm_ret=bm,
                                 multiplier='D')
    
    uploader.run()