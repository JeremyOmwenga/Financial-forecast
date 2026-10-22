import streamlit as st

st.set_page_config(page_title="Stock Forecast", layout="wide")

pages = [
    st.Page("stock_predict.py", title="Stock Forecast", icon=":material/trending_up:", default=True),
    st.Page("stock_dashboard.py", title="Portfolio & Savings", icon=":material/account_balance_wallet:"),
    st.Page("stock_details.py", title="Stock Details", icon=":material/insights:"),
    st.Page("dash_indicator.py", title="Real-Time Dashboard", icon=":material/monitoring:"),
]

st.navigation(pages).run()
