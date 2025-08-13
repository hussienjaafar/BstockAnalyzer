from pathlib import Path

import streamlit as st

def inject_custom_css() -> None:
    """Load style.css and inject it into the Streamlit app.

    This allows for a cleaner, more modern look without relying on Streamlit's
    internal class names. The CSS file lives next to this module and can be
    edited to tweak the application's visual theme.
    """
    css_path = Path(__file__).with_name("style.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

