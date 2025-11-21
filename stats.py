import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype
)
import requests
import os
import time
from tcia_utils import datacite
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as ET
import re
from collections import Counter
from bs4 import BeautifulSoup
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Publications - The Cancer Imaging Archive (TCIA)", layout="wide")

# Inject custom CSS to reduce padding
st.markdown(
    """
    <style>
    .stApp {
        margin-top: -50px;  /* Adjust this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# TTL and Cache Time Management
def should_refresh_cache(ttl: int = 86400):
    """
    Determine whether to refresh the cache based on a TTL (time-to-live).
    Default TTL is 1 day (86400 seconds).
    """
    cache_time_file = "cache_time.txt"

    # Check the timestamp of the last cache
    if not os.path.exists(cache_time_file):
        with open(cache_time_file, "w") as f:
            f.write(str(time.time()))
        return True  # No cache exists, so refresh

    with open(cache_time_file, "r") as f:
        last_cache_time = float(f.read().strip())

    # Check if TTL has expired
    if time.time() - last_cache_time > ttl:
        # Update the cache time
        with open(cache_time_file, "w") as f:
            f.write(str(time.time()))
        return True  # TTL expired, so refresh

    return False  # TTL not expired, no need to refresh

# Endnote XML Data Loader
@st.cache_data
def load_endnote_data():
    url = "https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml"
    response = requests.get(url)
    if response.status_code == 200:
        with open("Pubs_basedon_TCIA.xml", "wb") as f:
            f.write(response.content)

    endnote = parse_xml('Pubs_basedon_TCIA.xml')
    if 'electronic-resource-num' in endnote.columns:
        endnote['electronic-resource-num'] = endnote['electronic-resource-num'].str.strip()
    return endnote

# Datacite Data Loader
@st.cache_data
def load_data():
    """Load TCIA dataset information using datacite"""
    df = datacite.getDoi()
    return df

# Refresh cache based on TTL
if should_refresh_cache(ttl=86400):  # Set TTL to 1 day
    st.cache_data.clear()  # Clear cache for all functions

# helper functions for parsing the Endnote library to a dataframe
def replace_newline(s):
    """Replace newline characters in a string."""
    return s.replace('\r', '; ').replace('\n', '; ') if s is not None else ''

def get_text(element):
    """Get text from an XML element."""
    return element.text if element is not None else ''

def get_combined_text(element):
    """Get concatenated text from multiple nested XML elements."""
    if element is None:
        return ''
    return ''.join([text_element.text for text_element in element.findall('.//style') if text_element.text is not None])

# convert the endnote xml to a dataframe
def parse_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return pd.DataFrame()

    root = tree.getroot()
    records = []

    for item in root.findall('./records/record'):
        record = {
            'rec-number': replace_newline(get_text(item.find('rec-number'))),
            'ref-type': replace_newline(get_text(item.find('ref-type'))),
            'ref-type-name': item.find('ref-type').attrib.get('name') if item.find('ref-type') is not None else None,
            'authors': [replace_newline(get_combined_text(author)) for author in item.findall('contributors/authors/author')],
            'auth-address': replace_newline(get_combined_text(item.find('auth-address'))),
            'title': replace_newline(get_combined_text(item.find('titles/title'))),
            'secondary-title': replace_newline(get_combined_text(item.find('titles/secondary-title'))),
            'full-title': replace_newline(get_combined_text(item.find('periodical/full-title'))),
            'pages': replace_newline(get_combined_text(item.find('pages'))),
            'volume': replace_newline(get_combined_text(item.find('volume'))),
            'number': replace_newline(get_combined_text(item.find('number'))),
            'keywords': [replace_newline(get_combined_text(keyword)) for keyword in item.findall('keywords/keyword')],
            'year': replace_newline(get_combined_text(item.find('dates/year'))),
            'isbn': replace_newline(get_combined_text(item.find('isbn'))),
            'accession-num': replace_newline(get_combined_text(item.find('accession-num'))),
            'abstract': replace_newline(get_combined_text(item.find('abstract'))),
            'notes': replace_newline(get_combined_text(item.find('notes'))),
            'url': [replace_newline(get_combined_text(url)) for url in item.findall('urls/related-urls/url')],
            'electronic-resource-num': replace_newline(get_combined_text(item.find('electronic-resource-num'))),
            'remote-database-name': replace_newline(get_combined_text(item.find('remote-database-name')))
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters")
    if not modify:
        return df

    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            col_values = df[column]

            if col_values.apply(lambda x: isinstance(x, list)).any():
                col_values = col_values.apply(lambda x: tuple(x) if isinstance(x, list) else x)

            if is_categorical_dtype(col_values) or col_values.nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    col_values.unique(),
                    default=list(col_values.unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(col_values):
                _min = float(col_values.min())
                _max = float(col_values.max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(col_values):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        col_values.min(),
                        col_values.max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[col_values.astype(str).str.contains(user_text_input, flags=re.IGNORECASE, na=False)]

    return df

def create_app():

    # Sidebar for navigation
    with st.sidebar:
        st.title("Publications Based on TCIA")
        page = st.radio("Select a TCIA publication report", [
            "Verified TCIA Data Usage Citations",
            "DataCite Citations and Page Views",
            "TCIA Staff Publications"
        ])
        st.caption("Tip: Plots are interactive. Hover over plots to activate controls that will appear in the top right corner of each plot.  Tables can be exported to CSV files.")

    if st.query_params.get("clear_cache") == "true":
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
        st.query_params.clear()

    pubs_df = load_endnote_data()
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    if page == "Verified TCIA Data Usage Citations":
        st.subheader("Verified TCIA Data Usage Citations")
        st.markdown("We perform regular literature reviews in order to distinguish papers which explicitly analyzed TCIA datasets from those that simply mention TCIA or its data in some capacity.  You can download an Endnote XML file containing these verified citations [here](https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml).  This file should be usable as input to your favorite reference management system.")
        st.markdown("Publication years represent the year the paper was published, not the year we added it to our reference library.  There is generally a significant lag time between when papers are published and when we find time to add them to our library due to the amount of effort required to assess each paper and verify our data was actually used.  If you've analyzed TCIA data and don’t see your publication on this list please [notify us](https://www.cancerimagingarchive.net/support/)!")

        pubs_per_year = pubs_df['year'].value_counts().sort_index()
        cumulative_pubs = pubs_per_year.cumsum()

        df_totals = pd.DataFrame({
            'Publications per Year': pubs_per_year,
            'Cumulative Publications': cumulative_pubs
        }).T
        df_totals.index.name = 'Metric'

        fig = go.Figure()
        fig.add_trace(go.Bar(x=pubs_per_year.index, y=pubs_per_year.values, name='Yearly Citations'))
        fig.add_trace(go.Scatter(x=cumulative_pubs.index, y=cumulative_pubs.values, mode='lines+markers', name='Cumulative Citations', yaxis='y2'))
        fig.update_layout(
            xaxis_title='Year',
            yaxis=dict(title='Yearly Citations'),
            yaxis2=dict(title='Cumulative Citations', overlaying='y', side='right'),
            barmode='group',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig)

        st.dataframe(df_totals, use_container_width=True, hide_index=False)

        st.subheader("Explore Publications")
        st.markdown("Apply filters to our verified data usage citations and export subsets to CSV.  To export a CSV, mouse over the table and then use the download button in the top right corner.")

        filtered_endnote_explorer = filter_dataframe(pubs_df)
        st.dataframe(filtered_endnote_explorer)

        top_n_options = [10, 25, 50, 100]

        st.subheader(f'Top Authors by TCIA Publication Counts')
        col1, col2 = st.columns([1, 10])
        with col1:
            top_n_authors = st.selectbox('Select top N authors', options=top_n_options, index=2)

        all_authors = [author for sublist in filtered_endnote_explorer['authors'] for author in sublist]
        author_counts = Counter(all_authors)
        top_authors = pd.DataFrame(author_counts.most_common(top_n_authors), columns=['Author', 'Count'])
        fig_authors = px.bar(top_authors, x='Author', y='Count', title=f'Top {top_n_authors} Authors')
        fig_authors.update_xaxes(tickangle=45)
        st.plotly_chart(fig_authors)

        st.subheader('Reference Type Distribution')
        ref_type_counts = filtered_endnote_explorer['ref-type-name'].value_counts().reset_index()
        ref_type_counts.columns = ['Reference Type', 'Count']
        fig_ref_type = px.bar(ref_type_counts, x='Reference Type', y='Count', title='Reference Type Distribution')
        fig_ref_type.update_xaxes(tickangle=45)
        st.plotly_chart(fig_ref_type)

    elif page == "DataCite Citations and Page Views":
        st.title("DataCite Citations and Page Views")
        st.markdown("[DataCite](https://commons.datacite.org/repositories/nihnci.tcia) attempts to document all relationships between Digital Object Identifiers.  They also track metrics such as citation counts and page views as part of the [Make Data Count](https://makedatacount.org/) initiative. Statistics they've aggregated are summarized below, which includes information contributed by entities such as journals and other data repositories in addition to what TCIA submits.")
        st.markdown("Note: Page view tracking began on **2024-11-01**. Monthly averages for page views are calculated from this date (or the dataset creation date, whichever is later). Monthly averages for citations are calculated from the dataset creation date.")

        # --- Metric Calculations ---
        df['Created'] = pd.to_datetime(df['Created']).dt.tz_localize(None)
        
        PAGE_VIEW_START_DATE = pd.to_datetime('2024-11-01')
        CURRENT_DATE = pd.Timestamp('now')

        # Citations (All Time)
        df['MonthsExistence'] = (CURRENT_DATE.year - df['Created'].dt.year) * 12 + (CURRENT_DATE.month - df['Created'].dt.month) + 1
        df['MonthsExistence'] = df['MonthsExistence'].apply(lambda x: max(x, 1))
        df['CitationsMonthly'] = (df['CitationCount'] / df['MonthsExistence']).round(2)

        # Page Views (Since Nov 2024)
        df['EffectiveViewStart'] = df['Created'].apply(lambda x: max(x, PAGE_VIEW_START_DATE))
        df['MonthsTracked'] = (CURRENT_DATE.year - df['EffectiveViewStart'].dt.year) * 12 + (CURRENT_DATE.month - df['EffectiveViewStart'].dt.month) + 1
        df['MonthsTracked'] = df['MonthsTracked'].apply(lambda x: max(x, 1))
        df['PageViewsMonthly'] = (df['ViewCount'] / df['MonthsTracked']).round(2)

        # --- Display Table ---
        column_order = [
            "DOI", "Identifier", "ViewCount", "PageViewsMonthly", "CitationCount", "CitationsMonthly", 
            "ReferenceCount", "URL", "Title", "Related", "Created", "Updated", "Version", 
            "Rights", "RightsURI", "CreatorNames", "Description", "FundingReferences"
        ]
        available_cols = [col for col in column_order if col in df.columns]
        df_display = df[available_cols].copy()
        if "ViewCount" in df_display.columns:
            df_display = df_display.sort_values(by="ViewCount", ascending=False)
        df_display = df_display.reset_index(drop=True)

        st.subheader("Explore Page Views and Citation Counts")
        st.dataframe(filter_dataframe(df_display))

        # --- Treemaps ---
        col_treemap1, col_treemap2 = st.columns(2)

        with col_treemap1:
            # Treemap: Page Views
            # Note: We pass a LIST to hover_data to ensure strict index ordering in customdata
            fig_views = px.treemap(
                df,
                path=['Identifier'],
                values='PageViewsMonthly',
                title='Monthly Average Page Views by Dataset',
                hover_name='Title',
                hover_data=['ViewCount', 'MonthsTracked', 'CitationCount']
            )
            # Customdata mappings:
            # %{value} -> PageViewsMonthly
            # %{customdata[0]} -> ViewCount
            # %{customdata[1]} -> MonthsTracked
            # %{customdata[2]} -> CitationCount
            fig_views.update_traces(
                hovertemplate="<b>%{hovertext}</b><br><br>" +
                              "Avg Monthly Views: %{value}<br>" +
                              "Total Views: %{customdata[0]}<br>" +
                              "Months Tracked: %{customdata[1]}<br>" +
                              "Total Citations: %{customdata[2]}<extra></extra>"
            )
            st.plotly_chart(fig_views, use_container_width=True)

        with col_treemap2:
            # Treemap: Citations
            fig_citations = px.treemap(
                df,
                path=['Identifier'],
                values='CitationsMonthly',
                title='Monthly Average Citations by Dataset',
                hover_name='Title',
                hover_data=['CitationCount', 'MonthsExistence', 'ViewCount']
            )
            # Customdata mappings:
            # %{value} -> CitationsMonthly
            # %{customdata[0]} -> CitationCount
            # %{customdata[1]} -> MonthsExistence
            # %{customdata[2]} -> ViewCount
            fig_citations.update_traces(
                hovertemplate="<b>%{hovertext}</b><br><br>" +
                              "Avg Monthly Citations: %{value}<br>" +
                              "Total Citations: %{customdata[0]}<br>" +
                              "Months Since Creation: %{customdata[1]}<br>" +
                              "Total Views: %{customdata[2]}<extra></extra>"
            )
            st.plotly_chart(fig_citations, use_container_width=True)

        # --- Scatter Plot ---
        fig_scatter = px.scatter(
            df,
            x='ViewCount',
            y='CitationCount',
            hover_data=['Identifier', 'URL', 'Title'],
            title='Dataset Views vs Citations (Totals)',
            labels={'ViewCount': 'Total Views', 'CitationCount': 'Total Citations'}
        )
        if len(df) > 1:
            m, b = np.polyfit(df['ViewCount'], df['CitationCount'], 1)
            fig_scatter.add_trace(
                go.Scatter(
                    x=df['ViewCount'],
                    y=m * df['ViewCount'] + b,
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)


    elif page == "TCIA Staff Publications":
        st.subheader("TCIA Staff Publications")
        st.markdown("This page summarizes publications about TCIA authored by our team members.")
        WORDPRESS_URL = "https://www.cancerimagingarchive.net/publications-authored-by-tcia/"

        @st.cache_data
        def fetch_wordpress_page(url):
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = soup.find('ol')
                if content:
                    return str(content)
                else:
                    return "Could not find publications list."
            else:
                st.error(f"Failed to fetch page: {response.status_code}")
                return None

        page_content = fetch_wordpress_page(WORDPRESS_URL)
        if page_content:
            st.markdown(page_content, unsafe_allow_html=True)
        else:
            st.warning("Unable to fetch the page content.")

        col1, col2 = st.columns([10, 2])
        with col1:
            st.markdown(f"[View on cancerimagingarchive.net]({WORDPRESS_URL})", unsafe_allow_html=True)
        with col2:
            if st.button("🗑️ Clear Cache", help="Click to clear cached data"):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")

if __name__ == "__main__":
    create_app()