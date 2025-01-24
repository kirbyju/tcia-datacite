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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Publications - The Cancer Imaging Archive (TCIA)", layout="wide")

# Inject custom CSS to reduce padding
st.markdown(
    """
    <style>
    .stApp {
        margin-top: -75px;  /* Adjust this value as needed */
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
    endnote['electronic-resource-num'] = endnote['electronic-resource-num'].str.strip()
    return endnote

# Datacite Data Loader
@st.cache_data
def load_data():
    """Load TCIA dataset information using datacite"""
    df = datacite.getDoi()
    return df

# Function to fetch DICOM data
@st.cache_data
def load_dicom_downloads():
    #url = "https://cancerimagingarchive.net/downloads_dicom.csv"
    url = "https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/downloads_dicom.csv"
    df = pd.read_csv(url)

    # Remove any unnamed columns that might be causing issues
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Reshape the data
    df_long = pd.melt(
        df,
        id_vars=["Collection"],  # Keep 'Collection' as-is
        var_name="Date",         # Create a 'Date' column from column headers
        value_name="Bytes (GB)"  # Create a 'Bytes (GB)' column from values
    )

    # rename column
    df_long.rename(columns={"Bytes (GB)": "Downloads"}, inplace=True)

    # Convert 'Date' column to datetime -- YYYY-MM-DD
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")

    # Ensure required columns exist
    if 'Date' not in df_long.columns or 'Downloads' not in df_long.columns:
        st.error("Expected columns 'Date' and 'Downloads' not found in the data.")
        st.stop()

    # Drop rows with invalid or missing dates (if any)
    df_long = df_long.dropna(subset=["Date"])

    # Handle missing or invalid data in 'Downloads'
    df_long['Downloads'].fillna(0, inplace=True)  # Replace NaN with 0 for summation

    # Function to clean and convert the Downloads column
    def clean_downloads(value):
        try:
            # Remove commas and convert to float
            return float(value.replace(',', ''))
        except AttributeError:
            # If there are no commas, directly convert to float
            return float(value)

    # Apply the cleaning function to the Downloads
    df_long["Downloads"] = df_long["Downloads"].apply(clean_downloads)

    # Drop rows where 'Downloads' is 0
    df_long = df_long[df_long["Downloads"] != 0]

    return df_long

# function to fetch DICOM collection stats
@st.cache_data
def load_dicom_collection_report():
    # ingest summary CSV output of nbia.reportCollectionSummary(series, format = "csv")
    url = "https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/tcia_collection_report.csv"
    df_sizes = pd.read_csv(url)
    return df_sizes

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
    # Create element tree object
    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    # Get root element
    root = tree.getroot()

    # Create empty list for records
    records = []

    # Iterate through each record in the XML file
    for item in root.findall('./records/record'):
        # Create a dictionary to hold the data for this record
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
            #'url': replace_newline(get_combined_text(item.find('urls/related-urls/url'))),
            'electronic-resource-num': replace_newline(get_combined_text(item.find('electronic-resource-num'))),
            'remote-database-name': replace_newline(get_combined_text(item.find('remote-database-name')))
        }

        # Append the record to the list
        records.append(record)

    # Convert the list of records into a DataFrame
    df = pd.DataFrame(records)

    return df

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.

    Args:
        df (pd.DataFrame): Original dataframe.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Convert datetimes into a standard format (datetime, no timezone)
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
                # Handle list-like columns by converting to tuples
                col_values = col_values.apply(lambda x: tuple(x) if isinstance(x, list) else x)

            # Treat columns with < 10 unique values as categorical
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
    # Sidebar for navigation and logo
    with st.sidebar:
        # align vertical with main content using empty placeholders
        st.title("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        # Navigation selection
        page = st.radio("Select a TCIA publication report", [
            "Verified TCIA Data Usage Citations",
            "DataCite Citations and Page Views",
            "TCIA Staff Publications"
        ])

        st.caption("Tip: Plots are interactive. Hover over plots to activate controls that will appear in the top right corner of each plot.  Tables can be exported to CSV files.")

        # Add manual refresh button
        st.markdown("Source data refreshes daily, but can be manually updated using this button.")
        if st.button("Refresh All Data"):
            st.cache_data.clear()  # Clear cache for all functions

    # Main content area
    st.title("Publications Based on TCIA")
    st.markdown("Use the menu in the left sidebar to select a data source.")

    # Load endnote data
    pubs_df = load_endnote_data()

    # Load datacite data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Conditional rendering based on sidebar selection
    if page == "Verified TCIA Data Usage Citations":
        st.subheader("Verified TCIA Data Usage Citations")

        st.markdown("We perform regular literature reviews in order to distinguish papers which explicitly analyzed TCIA datasets from those that simply mention TCIA or its data in some capacity.  You can download an Endnote XML file containing these verified citations [here](https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml).  This file should be usable as input to your favorite reference management system.")
        st.markdown("If you've analyzed TCIA data and don’t see your publication on this list please [notify us](https://www.cancerimagingarchive.net/support/)!")

        # Count publications per year
        pubs_per_year = pubs_df['year'].value_counts().sort_index()
        # Calculate cumulative publications
        cumulative_pubs = pubs_per_year.cumsum()

        # Create a dataframe with yearly totals in columns
        df_totals = pd.DataFrame({
            'Publications per Year': pubs_per_year,
            'Cumulative Publications': cumulative_pubs
        }).T  # Transpose to make years column headers

        # Rename the index to make it more descriptive
        df_totals.index.name = 'Metric'

        # Create the chart
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
        # Show plot
        st.plotly_chart(fig)

        # Display dataframe below the chart
        st.dataframe(
            df_totals,
            use_container_width=True,
            hide_index=False
        )

        st.subheader("")
        st.markdown("Apply filters to our verified data usage citations and export subsets to CSV.")

        filtered_endnote_explorer = filter_dataframe(pubs_df)
        st.dataframe(filtered_endnote_explorer)

        # settings for dropdown menus that control how many authors/keywords in bar charts
        top_n_options = [10, 25, 50, 100]

        # Top N Authors
        st.subheader(f'Top Authors')
        top_n_authors = st.selectbox('Select top N authors', options=top_n_options)

        all_authors = [author for sublist in filtered_endnote_explorer['authors'] for author in sublist]
        author_counts = Counter(all_authors)
        top_authors = pd.DataFrame(author_counts.most_common(top_n_authors), columns=['Author', 'Count'])
        fig_authors = px.bar(top_authors, x='Author', y='Count', title=f'Top {top_n_authors} Authors')
        fig_authors.update_xaxes(tickangle=45)
        st.plotly_chart(fig_authors)

        # Reference Type Distribution
        st.subheader('Reference Type Distribution')
        ref_type_counts = filtered_endnote_explorer['ref-type-name'].value_counts().reset_index()
        ref_type_counts.columns = ['Reference Type', 'Count']
        fig_ref_type = px.bar(ref_type_counts, x='Reference Type', y='Count', title='Reference Type Distribution')
        fig_ref_type.update_xaxes(tickangle=45)
        st.plotly_chart(fig_ref_type)

        # count keywords for barchart and word cloud
        all_keywords = [keyword for sublist in filtered_endnote_explorer['keywords'] for keyword in sublist]
        keyword_counts = Counter(all_keywords)

        # Top N Keywords
        st.subheader(f'Top Keywords')
        top_n_keywords = st.selectbox('Select top N keywords', options=top_n_options)
        top_keywords = pd.DataFrame(keyword_counts.most_common(top_n_keywords), columns=['Keyword', 'Count'])
        fig_keywords = px.bar(top_keywords, x='Keyword', y='Count', title=f'Top {top_n_keywords} Keywords')
        fig_keywords.update_xaxes(tickangle=45)
        st.plotly_chart(fig_keywords)

        # Word Cloud for Keywords
        st.subheader('Keyword Cloud')
        wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(keyword_counts)

        # Create a Matplotlib figure with higher DPI
        plt.figure(figsize=(20, 10), dpi=300)  # Increase figure size and DPI
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Display the word cloud in Streamlit
        st.pyplot(plt, dpi=300)  # Set DPI for Streamlit display

    elif page == "DataCite Citations and Page Views":
        st.markdown("[DataCite](https://commons.datacite.org/repositories/nihnci.tcia) attempts to document all relationships between Digital Object Identifiers.")
        st.markdown("These reports include information contributed by other entities such as journals and other data repositories, as well as relationship data submitted by TCIA.")

        # Interactive table with citation details
        st.subheader("Page View and Citation Details")

        # Create a dataframe for display
        display_df = df[['Identifier', 'Title', 'CitationCount', 'ViewCount', 'URL', 'Rights']].copy()

        # Sort by view count by default
        display_df = display_df.sort_values('ViewCount', ascending=False)

        # Display interactive dataframe
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Bar chart comparing view counts for all datasets
        # Sort by view count (highest to lowest)
        sorted_by_views = df.sort_values('ViewCount', ascending=False)

        fig_views = px.bar(
            sorted_by_views,
            x='Identifier',
            y='ViewCount',
            title='Page Views by Dataset',
            labels={'Identifier': 'Dataset', 'ViewCount': 'Number of Views'},
            hover_data=['CitationCount', 'URL', 'Title']
        )
        fig_views.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_views, use_container_width=True)

        # Bar chart comparing citation counts for all datasets
        # Sort by citation count (highest to lowest)
        sorted_by_citations = df.sort_values('CitationCount', ascending=False)

        fig_citations = px.bar(
            sorted_by_citations,
            x='Identifier',
            y='CitationCount',
            title='Citations by Dataset',
            labels={'Identifier': 'Dataset', 'CitationCount': 'Number of Citations'},
            hover_data=['ViewCount', 'URL', 'Title']
        )
        fig_citations.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_citations, use_container_width=True)

        # Scatter plot comparing views vs citations
        fig = px.scatter(
            df,
            x='ViewCount',
            y='CitationCount',
            hover_data=['Identifier', 'URL', 'Title'],
            title='Dataset Views vs Citations',
            labels={'ViewCount': 'Number of Views', 'CitationCount': 'Number of Citations'}
        )

        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=df['ViewCount'],
                y=df['ViewCount'].map(lambda x: np.polyval(np.polyfit(df['ViewCount'], df['CitationCount'], 1), x)),
                name='Trend',
                line=dict(color='red', dash='dash')
            )
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed metadata filtering
        st.subheader("DataCite DOI Metadata")
        st.dataframe(filter_dataframe(df))

    elif page == "TCIA Staff Publications":
        st.subheader("TCIA Staff Publications")
        st.markdown("This page summarizes publications about TCIA authored by our team members.")
        # Confluence base URL and page ID
        CONFLUENCE_URL = "https://wiki.cancerimagingarchive.net"
        PAGE_ID = "52758446"  # Replace with your Confluence page ID

        @st.cache_data
        def fetch_public_confluence_page(page_id):
            url = f"{CONFLUENCE_URL}/rest/api/content/{page_id}"
            params = {
                "expand": "body.view",  # Use 'body.view' for public content
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch page: {response.status_code}")
                return None

        # Fetch and display the page
        page_data = fetch_public_confluence_page(PAGE_ID)
        if page_data:
            page_url = f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={PAGE_ID}"
            st.markdown(f"[View on Confluence]({page_url})", unsafe_allow_html=True)
            st.markdown(page_data["body"]["view"]["value"], unsafe_allow_html=True)
        else:
            st.warning("Unable to fetch the page content. Please check the page ID.")



if __name__ == "__main__":
    create_app()
