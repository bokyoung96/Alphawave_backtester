import logging

from uploader import *

logging.basicConfig(filename='streamlit_performance_uploader.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def run():
    """
    <DESCRIPTION>
    Run uploader for streamlit server connection.
    """
    st.set_page_config(layout='wide')
    st.markdown("<h1 style='text-align: center; color: black; font-size: 24px;'>Upload Data</h1>",
                unsafe_allow_html=True)

    with st.expander("Click here for file format guidelines"):
        st.markdown("""
        **Return data format guidelines:**
    - The file should be an Excel file (.xlsx).
    - The file should contain a sheet with the following columns:
      - `Date`: Date of the record (e.g., 2023-01-01)
      - `Portfolio`: Portfolio return (e.g., 0.01 for 1%)
      - `BM`: Benchmark return (Can be empty if no value available)
    - Example:
        ```
        Date       | Portfolio | BM
        -----------|-----------|-------
        2023-01-01 | 0.01      | 0.02
        2023-01-02 | 0.015     | 0.018
        ...
        ```

    **Caution:**
    - The `Portfolio` and `BM` indices must match.
    - The `BM` column must be present even if no data is available.

    ---
    
    **Weight data format guidelines:**
    - The file should be an Excel file (.xlsx).
    - The file should contain a sheet with the following columns:
      - `Date`: Date of the record (e.g., 2023-01-01)
      - `Ticker`: Ticker symbol (e.g., A005930, A000660)
      - `Weight`: Weight of each ticker (e.g., 0.5, NaN)
    - Example:
        ```
        Date       | A005930 | A000660
        -----------|---------|--------
        2023-01-01 | 0.5     | NaN
        2023-02-01 | 0.6     | 0.4
        ...
        ```

    **Caution:**
    - Only **long-only KOSPI stock analysis** is currently supported.
    - If the `skip` checkbox is checked, contribution analysis will not be performed.
    - The `Date` column should contain only the dates that are within the portfolio test period.
        """)

    uploaded_file = st.file_uploader("Upload return data", type="xlsx")

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            pf_ret = data[['Portfolio']]
            bm_ret = data[['BM']]

            performance_multiplier = st.selectbox('Select multiplier',
                                                  ['Y', 'M', 'D'],
                                                  help='Select the time frequency to adjust performance metrics')
            roll_multiplier = st.selectbox('Select rolling window',
                                           ['Y', '9M', '6M', '3M'],
                                           help='Select the size of the rolling window to adjust rolling performance metrics')

            weight_file = st.file_uploader("Upload weight data",
                                           type="xlsx",
                                           help="Upload weight data or check the skip option to proceed")
            skip_weight = st.checkbox('Skip weight data upload')

            weight_data = None
            start_date = None
            end_date = None
            contribution_multiplier = 'D'

            if not skip_weight:
                if weight_file is not None:
                    weight_data = pd.read_excel(weight_file, index_col=0)

                    min_date = pf_ret.index.min()
                    max_date = pf_ret.index.max()

                    start_date = st.date_input('Start Date',
                                               value=min_date,
                                               help=f'Select a start date (min: {min_date.date()}, max: {max_date.date()})')
                    end_date = st.date_input('End Date',
                                             value=max_date,
                                             help=f'Select an end date (min: {min_date.date()}, max: {max_date.date()})')
                    contribution_multiplier = st.selectbox('Select multiplier',
                                                           ['D', 'M', 'Y'],
                                                           help='Select the time frequency to adjust contribution metrics')

            logging.info("Uploaded file: %s",
                         uploaded_file.name)
            logging.info("File content:\n%s",
                         data.head().to_string())

            if skip_weight or weight_data is not None:
                uploader = StreamlitUploader(pf_ret=pf_ret,
                                             bm_ret=bm_ret,
                                             performance_multiplier=performance_multiplier,
                                             roll_multiplier=roll_multiplier,
                                             weight=weight_data,
                                             contribution_multiplier=contribution_multiplier,
                                             start_date=start_date,
                                             end_date=end_date)
                uploader.upload()

        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    run()
