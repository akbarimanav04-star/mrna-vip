"""Diagnostic: python debug_app.py   (run from the project folder, venv active)

Loads the dashboard headlessly and prints what it renders. Send the output if something looks wrong.
"""
import sys
from pathlib import Path

import streamlit
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parent
print("python   :", sys.version.split()[0])
print("streamlit:", streamlit.__version__)
print("folder   :", root)
at = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=300).run()
print("exceptions:", [e.value for e in at.exception])
print("warnings  :", [w.value for w in at.warning])
print("errors    :", [e.value for e in at.error])
print("sidebar   :", [c.value for c in at.sidebar.caption])
print("tabs      :", len(at.tabs))
print("metrics   :", {m.label: m.value for m in at.metric})
