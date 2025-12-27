#!/usr/bin/env python3
"""
Entry point for the Cryptocurrency Analysis Dashboard
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Get the path to the app
    app_path = os.path.join(os.path.dirname(__file__), "src", "app.py")
    
    # Run streamlit
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

