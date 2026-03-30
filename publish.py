"""
Plotly Cloud publish helper
===========================
Run this script to open the dashboard with Dash DevTools enabled.
A "Publish App" button will appear in the bottom-left corner of the
browser — click it, sign in to cloud.plotly.com, and deploy.

    python publish.py
"""
# Must be imported BEFORE app so that install_hook() registers the
# Plotly Cloud DevTools button into dash.hooks before dash.Dash() is created.
try:
    import plotly_cloud._devtool_hooks  # noqa: F401
except ImportError:
    print("Warning: plotly-cloud not installed. Publish button will not appear.")
    print("Install with: pip install plotly-cloud")

from app import app

if __name__ == "__main__":
    app.run(debug=True)
