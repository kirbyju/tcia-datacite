import streamlit as st
import pandas as pd
from tcia_utils import datacite

from pandas.api.types import (
    is_object_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype
)
import requests
import os
import time

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as ET
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Usage Statistics - The Cancer Imaging Archive (TCIA)", layout="wide")

def get_apa_citation(doi):
    url = f"https://citation.crosscite.org/format?doi={doi}&style=apa"
    headers = {"Accept": "text/plain"}  # Ensure plain text response
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Error: DOI not found or not available in AMA format."

# Endnote XML Data Loader
@st.cache_data(ttl=86400)
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
@st.cache_data(ttl=86400)
def load_datacite_data():
    """Load TCIA dataset information using datacite"""
    df = datacite.getDoi()
    return df

# function to fetch DICOM searches
@st.cache_data(ttl=86400)
def load_dicom_searches():
    df = pd.read_excel('https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/search_dicom_2025-04-07.xlsx')

    # Transpose the DataFrame
    transposed_df = df.transpose()  # or simply df.T

    # Reset index if needed (optional)
    transposed_df.reset_index(inplace=True)

    # Optionally, drop the 'index' column if it's no longer needed
    transposed_df = transposed_df.drop(columns=['index'])

    # Set the first row as the new column headers
    transposed_df.columns = transposed_df.iloc[0]  # Set the first row as the column headers
    transposed_df = transposed_df[1:]  # Drop the first row (since it's now the header row)

    # Convert all values in 'Search Criteria' to strings
    transposed_df['Search Criteria'] = transposed_df['Search Criteria'].astype(str)

    # Filter out rows based on updated conditions
    filtered_df = transposed_df[
        ~transposed_df['Search Criteria'].str.match('nan') &  # Remove rows where the value is NaN
        ~transposed_df['Search Criteria'].str.match(r'^[A-Za-z]$', na=False) &  # Remove single letters
        ~transposed_df['Search Criteria'].str.match(r'^\d(\.0)?$', na=False) &  # Remove single-digit integers or floats (e.g., 1 or 1.0)
        ~transposed_df['Search Criteria'].str.match(r'^[_-]$', na=False)  # Remove single underscores or dashes
    ]

    melted_df = filtered_df.melt(
        id_vars=["Search Criteria"],
        value_vars=["AnatomicalSite", "Collection", "Modality"],
        var_name="Category",
        value_name="Clicks"
    ).dropna()

    return melted_df

# Function to fetch DICOM downloads
@st.cache_data(ttl=86400)
def load_dicom_downloads():
    #url = "https://cancerimagingarchive.net/downloads_dicom.csv"
    url = "https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/downloads_dicom_2025-04-07.xlsx"
    df = pd.read_excel(url)

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
    df_long['Downloads'] = df_long['Downloads'].fillna(0)

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

    # TEMPORARY FIX Filter out rows where 'Collection' contains 'LDCT-and-Projection-data' AND 'Date' is in February or March of 2025
    df_long_filtered = df_long[
        ~((df_long['Collection'] == 'LDCT-and-Projection-data') &
          (df_long['Date'].dt.year == 2025) &
          (df_long['Date'].dt.month.isin([2, 3])))
    ]

    # TEMPORARY FIX -- revert to df_long once LDCT source data are corrected
    return df_long_filtered

# function to fetch DICOM collection stats to calculate sizes
@st.cache_data(ttl=86400)
def load_dicom_collection_report():
    # ingest summary CSV output of nbia.reportCollectionSummary(series, format = "csv")
    url = "https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/tcia_collection_report_2025-04-10.csv"
    df_sizes = pd.read_csv(url)
    return df_sizes

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

@st.cache_data(ttl=86400)
def load_and_process_aspera_data():
    # Read CSV, skip first row since it's junk
    file = "https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/downloads_aspera_2025-04-09.xlsx"
    df = pd.read_excel(file, skiprows=1)
    df['Unnamed: 0'] = df['Unnamed: 0'].ffill()

    # Rename unnamed first column to 'metric'
    df = df.rename(columns={df.columns[0]: 'metric'})

    # Melt the dataframe to convert date columns to rows
    melted_df = pd.melt(
        df,
        id_vars=['metric', 'Collection'],
        var_name='date',
        value_name='value'
    )

    # TEMPORARY FIX: Replace all occurrences of 'Bone-Marrow-Cytomorphology' in the entire DataFrame
    melted_df = melted_df.replace('Bone-Marrow-Cytomorphology', 'Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer')

    # Convert date strings to datetime objects -- Full MONTH YYYY format
    melted_df['date'] = pd.to_datetime(melted_df['date'])

    # Create separate dataframes for each metric
    complete_downloads = melted_df[melted_df['metric'] == 'Complete Downloads (Count)'].copy()
    complete_downloads_gb = melted_df[melted_df['metric'] == 'Complete Downloads (GB)'].copy()
    partial_downloads = melted_df[melted_df['metric'] == 'Partial Downloads (Count)'].copy()

    return complete_downloads, complete_downloads_gb, partial_downloads

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

def create_cumulative_visualization(df, title, y_axis_title):
    # Group by date and sum values across all collections
    daily_totals = df.groupby('date')['value'].sum().reset_index()
    # Calculate cumulative sum
    daily_totals['cumulative_value'] = daily_totals['value'].cumsum()

    # Create figure with secondary y-axis
    fig = px.bar(
        daily_totals,
        x='date',
        y='value',
        title=title,
        labels={
            'date': 'Date',
            'value': y_axis_title,
            'cumulative_value': f'Cumulative {y_axis_title}'
        }
    )

    # Add line trace for cumulative values
    line_trace = px.line(
        daily_totals,
        x='date',
        y='cumulative_value'
    ).data[0]

    # Update line trace to use secondary y-axis
    line_trace.yaxis = 'y2'
    fig.add_trace(line_trace)

    # Update layout for dual y-axes
    fig.update_layout(
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title=y_axis_title,
            side='left'
        ),
        yaxis2=dict(
            title=f'Cumulative {y_axis_title}',
            side='right',
            overlaying='y'
        )
    )

    # Update names for legend
    fig.data[0].name = 'Daily Total'
    fig.data[1].name = 'Cumulative Total'

    return fig

def create_single_collection_visualization(df, collection_name, title, y_axis_title):
    # Filter for selected collection
    collection_df = df[df['Collection'] == collection_name].copy()
    # Calculate cumulative sum for this collection
    collection_df['cumulative_value'] = collection_df['value'].cumsum()

    # Create figure with secondary y-axis
    fig = px.bar(
        collection_df,
        x='date',
        y='value',
        title=f"{title} - {collection_name}",
        labels={
            'date': 'Date',
            'value': y_axis_title,
            'cumulative_value': f'Cumulative {y_axis_title}'
        }
    )

    # Add line trace for cumulative values
    line_trace = px.line(
        collection_df,
        x='date',
        y='cumulative_value'
    ).data[0]

    # Update line trace to use secondary y-axis
    line_trace.yaxis = 'y2'
    fig.add_trace(line_trace)

    # Update layout for dual y-axes
    fig.update_layout(
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title=y_axis_title,
            side='left'
        ),
        yaxis2=dict(
            title=f'Cumulative {y_axis_title}',
            side='right',
            overlaying='y'
        )
    )

    # Update names for legend
    fig.data[0].name = 'Daily Total'
    fig.data[1].name = 'Cumulative Total'

    return fig

def create_app():
    # Load endnote data
    pubs_df = load_endnote_data()

    # Load datacite data
    try:
        df = load_datacite_data()
        #st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # load and process aspera downloads
    complete_downloads, complete_downloads_gb, partial_downloads = load_and_process_aspera_data()
    #st.dataframe(complete_downloads)

    # Load DICOM search metrics
    try:
        dicom_search_df = load_dicom_searches()
    except Exception as e:
        st.error(f"Failed to load DICOM search metrics: {e}")
        st.stop()

    # Load DICOM download metrics
    try:
        dicom_df = load_dicom_downloads()
    except Exception as e:
        st.error(f"Failed to load DICOM download metrics: {e}")
        st.stop()

    # Load DICOM collection size metrics for calculating 'normalized' download stats
    try:
        collection_size_df = load_dicom_collection_report()

        # Drop all columns except "Collection" and "File Size"
        collection_size_df = collection_size_df[['Collection', 'File Size']]

        # Convert 'File Size' from bytes to GBytes
        collection_size_df['File Size'] = collection_size_df['File Size'] / (1024**3)

    except Exception as e:
        st.error(f"Failed to load DICOM collection collection report data: {e}")
        st.stop()


    # Sidebar for navigation and logo
    with st.sidebar:
        st.image("https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-01.png", use_container_width=True)

        # Navigation selection with optional use of URL query parameters
        # Ensure the 'Identifier' column exists and get the unique values
        unique_identifiers = sorted(df['Identifier'].unique().tolist())

        # Get query parameters from the URL
        query_params = st.query_params
        selected_id = query_params.get("dataset", [None])

        # Ensure the query parameter value exists in unique_identifiers, else default to the first item
        default_index = unique_identifiers.index(selected_id) if selected_id in unique_identifiers else 0

        # Create a dropdown menu with the unique values
        dataset = st.selectbox(
            "Select a Dataset",
            unique_identifiers,
            index=default_index
        )

        st.caption("Tip: Plots are interactive. Hover over plots to activate controls that will appear in the top right corner of each plot.  Tables can be exported to CSV files.")

        st.markdown("Last updated: 2025-04-04")

        # Add manual refresh button
        #st.markdown("Source data refreshes daily, but can be manually updated using this button.")
        #if st.button("Refresh All Data"):
        #    st.cache_data.clear()  # Clear cache for all functions

    # Main content area
    st.title(f"Data Usage Statistics: {dataset}")

    # Extract the DOI value where Identifier equals selected_id
    doi = df.loc[df['Identifier'] == dataset, 'DOI'].values[0]

    # get full DOI citation from crossref
    if doi:
        citation = get_apa_citation(doi)

        # Display citation
        st.subheader("Citation:")
        st.info(f"{citation}")
        st.markdown("Please remember to always include the full dataset citation in your publication references to help ensure accurate usage metrics.")

    st.subheader("Page Views")
    st.markdown("We leverage DataCite's Make Data Count initiative to track page views of our datasets.  See how yours stacks up to the others in TCIA using the zoom and pan functions for the plot below.  These controls appear in the top right corner of the plot after you mouse over it.  Your dataset will appear at the top of the list to make it easier to find.")

    # Create a copy of the original DataFrame
    page_views_df = df.copy()

    # Filter out rows with zero ViewCount
    page_views_df = page_views_df[page_views_df['ViewCount'] > 0]

    # Sort the DataFrame by ViewCount in descending order
    page_views_df = page_views_df.sort_values(by='ViewCount', ascending=False)

    # If the dataset value is not in the Identifiers, handle gracefully
    if dataset not in page_views_df['Identifier'].values:
        dataset = page_views_df['Identifier'].iloc[0]  # Default to the first Identifier if not found

    # Create a Streamlit select box pre-populated with the 'dataset' value
    selected_identifier = st.selectbox(
        "Search for an Identifier:",
        page_views_df['Identifier'],
        index=page_views_df['Identifier'].tolist().index(dataset)  # Pre-select the dataset value
    )

    # Highlight the selected Identifier in the chart
    highlighted_data = page_views_df.copy()
    highlighted_data['Highlight'] = highlighted_data['Identifier'].apply(
        lambda x: 'Selected' if x == selected_identifier else 'Others'
    )

    fig = px.bar(
        highlighted_data,
        x='ViewCount',
        y='Identifier',
        orientation='h',
        text='ViewCount',
        color='Highlight',  # Differentiate the selected Identifier visually
        hover_data=['Title', 'URL']
    )

    fig.update_layout(
        title='Page views by Dataset',
        xaxis_title='ViewCount',
        yaxis_title='Dataset Identifier',
        yaxis=dict(showticklabels=True, automargin=True),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Verified TCIA Data Usage Citations")

    st.markdown("We perform regular literature reviews in order to distinguish papers which explicitly analyzed TCIA datasets from those that simply mention TCIA or its data in some capacity.  You can download an Endnote XML file containing these verified citations [here](https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml).  This file should be usable as input to your favorite reference management system.")
    st.markdown("Publication years represent the year the paper was published, not the year we added it to our reference library.  There is generally a significant lag time between when papers are published and when we find time to add them to our library due to the amount of effort required to assess each paper and verify TCIA data was actually used.  If you're aware of additional publications that should be on this list, please [notify us](https://www.cancerimagingarchive.net/support/)!")

    # Function to filter rows where 'keywords' contain the dataset value
    def filter_by_keyword(df, keyword):
        return df[df['keywords'].apply(lambda x: keyword in x)]

    # Filter the DataFrame
    pubs_df = filter_by_keyword(pubs_df, dataset)

    if pubs_df.empty:
        st.warning("There are no Verified Citations that we're aware of using this dataset. Please contact us if there are any you'd like to add!")
    else:
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

        # Function to determine appropriate tick intervals
        def determine_tick_interval(values):
            values = [int(value) for value in values if value is not None and value != '']
            if len(values) < 2:
                return 1
            range_values = max(values) - min(values)
            return max(1, range_values // 5)

        # Create the chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pubs_per_year.index, y=pubs_per_year.values, name='Yearly Citations'))
        fig.add_trace(go.Scatter(x=cumulative_pubs.index, y=cumulative_pubs.values, mode='lines+markers', name='Cumulative Citations', yaxis='y2'))

        # Determine tick intervals for years and citations
        x_tick_interval = determine_tick_interval(pubs_per_year.index)
        y_tick_interval = determine_tick_interval(pubs_per_year.values)
        y2_tick_interval = determine_tick_interval(cumulative_pubs.values)

        # Update layout to show appropriate ticks
        fig.update_layout(
            xaxis=dict(
                title='Year',
                tickmode='linear',
                dtick=x_tick_interval
            ),
            yaxis=dict(
                title='Yearly Citations',
                tickmode='linear',
                dtick=y_tick_interval
            ),
            yaxis2=dict(
                title='Cumulative Citations',
                overlaying='y',
                side='right',
                tickmode='linear',
                dtick=y2_tick_interval
            ),
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

        st.markdown("Apply filters to our verified data usage citations and export subsets to CSV.  To export a CSV, mouse over the table and then use the download button in the top right corner.")

        filtered_endnote_explorer = filter_dataframe(pubs_df)
        st.dataframe(filtered_endnote_explorer)

        # settings for dropdown menus that control how many authors/keywords in bar charts
        top_n_options = [10, 25, 50, 100]

        # Top N Authors
        col1, col2 = st.columns([1, 8])
        with col1:
            top_n_authors = st.selectbox('Select top N authors', options=top_n_options, index=1)

        all_authors = [author for sublist in filtered_endnote_explorer['authors'] for author in sublist]
        author_counts = Counter(all_authors)
        top_authors = pd.DataFrame(author_counts.most_common(top_n_authors), columns=['Author', 'Count'])
        fig_authors = px.bar(top_authors, x='Author', y='Count', title=f'Top {top_n_authors} Authors of Verified Publications')
        fig_authors.update_xaxes(tickangle=45)
        st.plotly_chart(fig_authors)

        # Reference Type Distribution
        ref_type_counts = filtered_endnote_explorer['ref-type-name'].value_counts().reset_index()
        ref_type_counts.columns = ['Reference Type', 'Count']
        fig_ref_type = px.bar(ref_type_counts, x='Reference Type', y='Count', title='Reference Type Distribution')
        fig_ref_type.update_xaxes(tickangle=45)
        st.plotly_chart(fig_ref_type)

    # dicom search stats
    #st.subheader("DICOM Searches Over Time (filter clicks)")

    #fig = px.treemap(
    #    filtered_df,
    #    path=["Category", "Search Criteria"],
    #    values="Clicks",
    #    title="Search Criteria Popularity by Category"
    #)

    #st.plotly_chart(fig, use_container_width=True)

    # dicom download stats
    st.subheader("DICOM Downloads Over Time (GBytes)")

    dicom_collections = sorted(dicom_df["Collection"].unique().tolist())

    # Make sure selected dataset has dicom downloads
    if dataset in dicom_collections:
        filtered_df = dicom_df[dicom_df["Collection"] == dataset]

        # Group by date and sum the downloads
        grouped_df = filtered_df.groupby('Date').agg({'Downloads': 'sum'}).reset_index()

        # ensure Date is datetime format
        grouped_df['Date'] = pd.to_datetime(grouped_df['Date'])

        # Filter out rows where 'Downloads' is zero
        grouped_df = grouped_df[grouped_df['Downloads'] > 0]

        # Create two columns
        col1, col2 = st.columns([2.5, 1.5])
        with col1:
            st.markdown("Since TCIA gives users the option to download partial datasets, we provide here a Normalized Plot that divides the total GBytes downloaded by the size of the Collection to estimate the number of times the full dataset equivalent was downloaded.")
            # Add a radio button to select plot type
            plot_type = st.radio(
                "Choose plot type:",
                ("Normalized Downloads",
                    "Total Downloads")
                )

        with col2:
            # Create slider bar
            min_date = grouped_df['Date'].min().to_pydatetime()
            max_date = grouped_df['Date'].max().to_pydatetime()

            # Streamlit slider
            start_date, end_date = st.slider(
                "Select Date Range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

            # Filter the data for the selected date range
            filtered_dataset_df = grouped_df[
                (grouped_df['Date'] >= start_date) &
                (grouped_df['Date'] <= end_date)
            ]

        # plot based on radio button selection
        if plot_type == "Total Downloads":
            fig_time = px.bar(
                filtered_dataset_df,
                x='Date',
                y='Downloads',
                title=f'Selected Data - {dataset}',
                labels={'Date': 'Date', 'Downloads': 'Downloads (GBytes)'}
            )

            fig_time.update_xaxes(
                tickangle=45,
                tickmode='auto',
                nticks=20  # Adjust based on your data
            )

            fig_time.update_layout(
                margin=dict(b=80)  # Add some bottom margin for rotated labels
            )

            st.plotly_chart(fig_time, use_container_width=True)

        else:
            # Calculate normalized downloads
            #filtered_dataset_df['Normalized Downloads'] = filtered_dataset_df['Downloads'] / filtered_dataset_df['File Size']
            # Get File Size for the selected dataset
            file_size = collection_size_df.loc[collection_size_df['Collection'] == dataset, 'File Size']

            if file_size.empty:
                st.error(f"File Size not found for dataset: {dataset}")
            else:
                file_size = file_size.iloc[0]  # Extract the single value

                # Calculate normalized downloads without merging
                filtered_dataset_df['Normalized Downloads'] = filtered_dataset_df['Downloads'] / file_size

            # Group data by dataset and sort
            fig_time = px.bar(
                filtered_dataset_df,
                x='Date',
                y='Normalized Downloads',
                title=f'Selected Data - {dataset}',
                labels={'Date': 'Date', 'Normalized Downloads': 'Normalized Downloads'}
            )

            fig_time.update_xaxes(
                tickangle=45,
                tickmode='auto',
                nticks=20  # Adjust based on your data
            )

            fig_time.update_layout(
                margin=dict(b=80)  # Add some bottom margin for rotated labels
            )

            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.warning(f'This dataset does not have any DICOM Downloads: {dataset}')

    # Aspera download status
    st.subheader('Non-DICOM (Aspera) Downloads Over Time')
    if complete_downloads is not None:
        # Create two columns with different widths
        col1, col2 = st.columns([2.5, 1.5])
        with col1:
            # Create visualization selector
            viz_type = st.radio(
                "Select visualization type:",
                # exclude complete download GB chart for now since we don't have partial equivalent
                #["Complete Downloads", "Download Volume (GB)", "Partial Downloads"]
                ["Complete Downloads", "Partial Downloads"]
            )

            # Get the appropriate dataframe based on selection
            if viz_type == "Complete Downloads":
                df_to_use = complete_downloads
                y_axis_title = "Number of Downloads"
            elif viz_type == "Download Volume (GB)":
                df_to_use = complete_downloads_gb
                y_axis_title = "Download Size (GB)"
            else:
                df_to_use = partial_downloads
                y_axis_title = "Number of Partial Downloads"

        aspera_collections = sorted(df_to_use["Collection"].unique().tolist())

        if dataset in aspera_collections:
            fig = create_single_collection_visualization(
                df_to_use,
                dataset,
                f"{viz_type}",
                y_axis_title
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f'This dataset does not have any Non-DICOM Downloads: {dataset}')

if __name__ == "__main__":
    create_app()
