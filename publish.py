"""
Plotly Cloud publish helper
===========================
Run this script to open the dashboard with Dash DevTools enabled.
A "Publish App" button will appear in the bottom-right corner of the
browser — click it, sign in to cloud.plotly.com, and deploy.

    python publish.py
"""
from app import app

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
