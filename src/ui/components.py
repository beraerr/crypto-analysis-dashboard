"""
Reusable UI components for the Streamlit application
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from src.utils.config import config
from src.utils.helpers import format_currency, format_percentage, calculate_percentage_change


def render_header():
    """Render the main header"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <strong>DISCLAIMER:</strong> This application is for educational and analysis purposes only. 
        The information provided does not constitute investment advice. Cryptocurrency investments carry high risk. 
        Make investment decisions at your own risk.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Cryptocurrency Price Analysis & Technical Indicators</h1>', 
                unsafe_allow_html=True)


def render_price_metrics(data: pd.DataFrame) -> None:
    """Render price metrics in columns"""
    latest_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2] if len(data) > 1 else latest_price
    price_change, price_change_pct = calculate_percentage_change(previous_price, latest_price)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=format_currency(latest_price),
            delta=format_percentage(price_change_pct)
        )
    
    with col2:
        st.metric(
            label="24h Change",
            value=format_currency(price_change),
            delta=format_percentage(price_change_pct)
        )
    
    with col3:
        st.metric(
            label="Daily High",
            value=format_currency(data['High'].iloc[-1])
        )
    
    with col4:
        st.metric(
            label="Daily Low",
            value=format_currency(data['Low'].iloc[-1])
        )


def render_news_card(news: Dict, sentiment: Dict) -> None:
    """Render a single news article card"""
    polarity = sentiment.get('polarity', 0.0)
    
    if polarity > 0.1:
        border_color = "4px solid #28a745"
        sentiment_label = "Positive"
    elif polarity < -0.1:
        border_color = "4px solid #dc3545"
        sentiment_label = "Negative"
    else:
        border_color = "4px solid #6c757d"
        sentiment_label = "Neutral"
    
    card_html = f"""
    <div style="
        border-left: {border_color};
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="margin-top: 0;">
            {news['title']}
        </h4>
        <p style="color: #6c757d; font-size: 0.9rem; margin-bottom: 0.5rem;">
            <strong>Source:</strong> {news['source']} | 
            <strong>Language:</strong> {'TR' if news.get('language') == 'tr' else 'EN'} | 
            <strong>Weight:</strong> {news.get('weight', 1.0):.1f}x | 
            <strong>Sentiment:</strong> {sentiment_label} ({polarity:.2f})
        </p>
        <p style="font-size: 0.85rem; color: #495057;">
            {news.get('summary', '')[:200]}...
        </p>
        <a href="{news['link']}" target="_blank" style="color: #007bff; text-decoration: none;">
            Read Article →
        </a>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_fear_greed_index(fng_data: Dict) -> None:
    """Render Fear & Greed Index"""
    if fng_data.get('success'):
        value = fng_data['value']
        classification = fng_data['classification']
        
        st.metric(
            label="Current Value",
            value=str(value),
            delta=classification
        )
        
        st.progress(value / 100)
        st.caption(f"**Classification:** {classification}")
        
        if value >= 75:
            st.warning("Extreme Greed: Market is overly optimistic. Exercise caution!")
        elif value <= 25:
            st.info("Extreme Fear: Market is overly pessimistic. Potential opportunity!")
    else:
        st.info("Fear & Greed Index data is currently unavailable.")


def render_sentiment_summary(sentiment_result: Dict) -> None:
    """Render sentiment analysis summary"""
    score = sentiment_result.get('score', 0.0)
    sentiment_color = 'green' if score > 0.1 else 'red' if score < -0.1 else 'gray'
    
    st.markdown(f"**Overall Sentiment:** {sentiment_result.get('message', 'N/A')}")
    
    col_pos, col_neg, col_neut = st.columns(3)
    with col_pos:
        st.metric("Positive", sentiment_result.get('positive_count', 0))
    with col_neg:
        st.metric("Negative", sentiment_result.get('negative_count', 0))
    with col_neut:
        st.metric("Neutral", sentiment_result.get('neutral_count', 0))
    
    st.metric(
        "Weighted Sentiment Score",
        f"{score:.3f}",
        delta=f"{'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'}"
    )


def render_prediction_table(future_dates: pd.DatetimeIndex, predictions: np.ndarray, 
                           model_name: str) -> None:
    """Render prediction results table"""
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predictions
    })
    
    st.dataframe(
        pred_df.style.format({
            'Predicted Price': '${:,.2f}'
        }),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🤖 **Model Used:** {model_name}")
    with col2:
        if len(predictions) > 0:
            st.metric(
                "Predicted Price",
                format_currency(predictions[-1])
            )

