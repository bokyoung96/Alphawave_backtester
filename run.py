import logging

from uploader import *

# NOTE: FOR MAC, streamlit run app.py --server.enableXsrfProtection false

logging.basicConfig(filename='streamlit_performance_uploader.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def run():
    """
    <DESCRIPTION>
    Run uploader for streamlit server connection.
    """
    st.set_page_config(layout='wide')
    st.markdown("<h1 style='text-align: center; color: black; font-size: 24px;'>Upload Return Data</h1>",
                unsafe_allow_html=True)

    with st.expander("Click here for file format guidelines"):
        st.markdown("""
        **File format guidelines:**
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
    - Turn off `dark mode` for better visualization.
        """)

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            pf_ret = data[['Portfolio']]
            bm_ret = data[['BM']]

            logging.info("Uploaded file: %s",
                         uploaded_file.name)
            logging.info("File content:\n%s",
                         data.head().to_string())

            multiplier = st.selectbox('Select Multiplier',
                                      ['Y', 'M', 'D'],
                                      help='Select the time frequency to adjust performance metrics')
            roll_multiplier = st.selectbox('Select Rolling Window',
                                           ['Y', '9M', '6M', '3M'],
                                           help='Select the size of the rolling window to adjust rolling performance metrics')

            uploader = StreamlitUploader(pf_ret=pf_ret,
                                         bm_ret=bm_ret,
                                         multiplier=multiplier,
                                         roll_multiplier=roll_multiplier)
            uploader.upload()
        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    run()
