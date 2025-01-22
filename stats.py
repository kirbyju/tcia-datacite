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

st.set_page_config(page_title="TCIA Data Usage Statistics", layout="wide")

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
        # Logo at the top of the sidebar
        st.image("https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-01.png", use_container_width=True)

        # Navigation selection
        page = st.radio("Select View", [
            "Dataset Citations by Year",
            "Citations and Page Views by Dataset",
            "Downloads (DICOM Data)",
            "Downloads (Non-DICOM Data)",
            "Endnote Citation Explorer",
            "DataCite Metadata Explorer"
        ])

        # Add manual refresh button
        st.markdown("Source data refreshes daily, but can be manually updated using this button.")
        if st.button("Refresh All Data"):
            st.cache_data.clear()  # Clear cache for all functions

    # Main content area
    st.title("TCIA Data Usage Statistics")
    st.markdown("Explore a variety of data usage metrics for TCIA datasets.  Select from the available reports in the left sidebar.")

    st.caption("Tip: Plots are interactive. Controls can be found at the top right corner of each plot.  Tables can be exported to CSV files.")

    # Load endnote data
    pubs_df = load_endnote_data()

    # Load datacite data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Conditional rendering based on sidebar selection
    if page == "Dataset Citations by Year":
        st.markdown("You can download an Endnote XML file containing all of our known citations of TCIA datasets [here](https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml).")

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

        st.subheader("Dataset Citations Over Time")
        st.metric("Total Citations", len(pubs_df))
        # Create the chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pubs_per_year.index, y=pubs_per_year.values, name='Publications per Year'))
        fig.add_trace(go.Scatter(x=cumulative_pubs.index, y=cumulative_pubs.values, mode='lines+markers', name='Cumulative Publications', yaxis='y2'))
        fig.update_layout(
            xaxis_title='Year',
            yaxis=dict(title='Number of Citations'),
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

    elif page == "Citations and Page Views by Dataset":

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

    elif page == "Endnote Citation Explorer":
        st.subheader("Endnote Citations")
        st.markdown("You can download an Endnote XML file containing all of our known citations of TCIA datasets [here](https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml).")
        st.markdown("This view provides an interactive filtering mechanism to search its contents and export subsets to CSV.")

        filtered_endnote_explorer = filter_dataframe(pubs_df)
        st.dataframe(filtered_endnote_explorer)

        # count keywords for barchart and word cloud
        all_keywords = [keyword for sublist in filtered_endnote_explorer['keywords'] for keyword in sublist]
        keyword_counts = Counter(all_keywords)

        # Word Cloud for Keywords
        st.subheader('Keyword Word Cloud')
        wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(keyword_counts)

        # Create a Matplotlib figure with higher DPI
        plt.figure(figsize=(20, 10), dpi=300)  # Increase figure size and DPI
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Display the word cloud in Streamlit
        st.pyplot(plt, dpi=300)  # Set DPI for Streamlit display

        # settings for dropdown menus that control how many authors/keywords in bar charts
        top_n_options = [10, 25, 50, 100]

        # Top N Keywords
        st.subheader(f'Top Keywords')
        top_n_keywords = st.selectbox('Select top N keywords', options=top_n_options)
        top_keywords = pd.DataFrame(keyword_counts.most_common(top_n_keywords), columns=['Keyword', 'Count'])
        fig_keywords = px.bar(top_keywords, x='Keyword', y='Count', title=f'Top {top_n_keywords} Keywords')
        fig_keywords.update_xaxes(tickangle=45)
        st.plotly_chart(fig_keywords)

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

        # Year Distribution
        st.subheader('Year Distribution')
        year_counts = filtered_endnote_explorer['year'].value_counts().reset_index()
        year_counts.columns = ['Year', 'Count']
        fig_year = px.bar(year_counts, x='Year', y='Count', title='Year Distribution')
        fig_year.update_xaxes(tickangle=45)
        st.plotly_chart(fig_year)

    elif page == "DataCite Metadata Explorer":
        st.subheader("DataCite DOI Metadata")
        st.dataframe(filter_dataframe(df))

    elif page == "Downloads (Non-DICOM Data)":

        @st.cache_data
        def load_and_process_aspera_data(file):
            # Read CSV, skip first row since it's junk
            df = pd.read_excel(file, skiprows=1)

            # Rename unnamed first column to 'metric'
            df = df.rename(columns={df.columns[0]: 'metric'})

            # Melt the dataframe to convert date columns to rows
            melted_df = pd.melt(
                df,
                id_vars=['metric', 'Collection'],
                var_name='date',
                value_name='value'
            )

            # Convert date strings to datetime objects
            melted_df['date'] = pd.to_datetime(melted_df['date'])

            # Create separate dataframes for each metric
            complete_downloads = melted_df[melted_df['metric'] == 'Complete Downloads (Count)'].copy()
            complete_downloads_gb = melted_df[melted_df['metric'] == 'Complete Downloads (GB)'].copy()
            partial_downloads = melted_df[melted_df['metric'] == 'Partial Downloads (Count)'].copy()

            return complete_downloads, complete_downloads_gb, partial_downloads

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

        # load and process aspera data
        uploaded_file = "https://github.com/kirbyju/tcia-datacite/raw/refs/heads/main/downloads_aspera.xlsx"
        complete_downloads, complete_downloads_gb, partial_downloads = load_and_process_aspera_data(uploaded_file)

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
            with col2:
                # Add a selection box for collection
                collections = sorted(df_to_use["Collection"].unique().tolist())
                collections.insert(0, "All Collections")
                selected_collection = st.selectbox("Select a Collection", collections)

            if selected_collection == "All Collections":
                fig = create_cumulative_visualization(
                    df_to_use,
                    f"{viz_type} - All Collections",
                    y_axis_title
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = create_single_collection_visualization(
                    df_to_use,
                    selected_collection,
                    f"{viz_type}",
                    y_axis_title
                )
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Downloads (DICOM Data)":
        st.subheader("DICOM Downloads Over Time (GBytes)")

        # Load DICOM download metrics
        try:
            dicom_df = load_dicom_downloads()
        except Exception as e:
            st.error(f"Failed to load DICOM download metrics: {e}")
            st.stop()

        # Load DICOM collection size metrics
        try:
            collection_size_df = load_dicom_collection_report()

            # Drop all columns except "Collection" and "File Size"
            collection_size_df = collection_size_df[['Collection', 'File Size']]

            # Convert 'File Size' from bytes to GBytes
            collection_size_df['File Size'] = collection_size_df['File Size'] / (1024**3)

            # Merge the DICOM data with collection sizes
            dicom_df = pd.merge(dicom_df, collection_size_df, on='Collection', how='left')

        except Exception as e:
            st.error(f"Failed to load DICOM collection collection report data: {e}")
            st.stop()

        # Create two columns with different widths
        col1, col2 = st.columns([2.5, 1.5])
        with col1:
            st.markdown("Display aggregate totals or choose an individual collection.")
            st.markdown("Note that in 2024 Q1 there is a known gap in reporting data.")
        with col2:
            # Add a selection box for collection
            collections = sorted(dicom_df["Collection"].unique().tolist())
            collections.insert(0, "All Collections")
            selected_collection = st.selectbox("Select a Collection", collections)

        # Filter data based on selection
        if selected_collection == "All Collections":
            filtered_df = dicom_df
        else:
            filtered_df = dicom_df[dicom_df["Collection"] == selected_collection]

        # Group by date and sum the downloads
        grouped_df = filtered_df.groupby('Date').agg({'Downloads': 'sum'}).reset_index()

        # ensure Date is datetime format
        grouped_df['Date'] = pd.to_datetime(grouped_df['Date']).dt.strftime('%Y-%m-%d')

        # Filter out rows where 'Downloads' is zero
        grouped_df = grouped_df[grouped_df['Downloads'] > 0]

        fig_time = px.bar(
            grouped_df,
            x='Date',
            y='Downloads',
            title=f'Selected Data - {selected_collection}',
            labels={'Date': 'Date', 'Downloads': 'Total Number of Downloads'}
        )

        fig_time = px.bar(
            grouped_df,
            x='Date',
            y='Downloads',
            title=f'Selected Data - {selected_collection}',
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

        # Dataset-level analysis with optional date filtering
        if 'Collection' in dicom_df.columns:
            # Add a date range filter for the dataset-level analysis
            st.subheader("DICOM Collection Popularity by Downloads")

            # Create two columns with different widths
            col1, col2 = st.columns([2.5, 1.5])
            with col1:
                st.markdown("The Normalized plot divides the total GBytes downloaded by the size of the Collection to prevent larger Collections from skewing the comparison.")
                # Add a radio button to select plot type
                plot_type = st.radio(
                    "Choose plot type:",
                    ("Normalized Downloads",
                        "Total Downloads")
                    )

            with col2:
                # Create slider bar
                min_date = dicom_df['Date'].min()
                max_date = dicom_df['Date'].max()

                # Streamlit slider
                start_date, end_date = st.slider(
                    "Select Date Range:",
                    min_value=min_date.to_pydatetime(),  # Convert to Python datetime
                    max_value=max_date.to_pydatetime(),
                    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                    format="YYYY-MM-DD",
                )

                # Filter the data for the selected date range
                filtered_dataset_df = dicom_df[
                    (dicom_df['Date'] >= pd.Timestamp(start_date)) &
                    (dicom_df['Date'] <= pd.Timestamp(end_date))
                ]

            # plot based on radio button selection
            if plot_type == "Total Downloads":
                # Group data by dataset and sort
                dataset_downloads = filtered_dataset_df.groupby('Collection', as_index=False)['Downloads'].sum().sort_values(by='Downloads', ascending=False)
                # Create bar chart for dataset-level visualization
                fig_dataset = px.bar(
                    dataset_downloads,
                    x='Collection',
                    y='Downloads',
                    title=f'Total Downloads by Collection (Filtered: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
                    labels={'Collection': 'Collection', 'Downloads': 'Total Downloads'},
                    text='Downloads'
                )
                fig_dataset.update_xaxes(tickangle=45)
                fig_dataset.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                st.plotly_chart(fig_dataset, use_container_width=True)
            else:
                # Calculate normalized downloads
                filtered_dataset_df['Normalized Downloads'] = filtered_dataset_df['Downloads'] / filtered_dataset_df['File Size']
                # Group data by dataset and sort
                dataset_downloads = filtered_dataset_df.groupby('Collection', as_index=False)['Normalized Downloads'].sum().sort_values(by='Normalized Downloads', ascending=False)
                # Create bar chart for dataset-level visualization
                fig_dataset = px.bar(
                    dataset_downloads,
                    x='Collection',
                    y='Normalized Downloads',
                    title=f'Normalized Downloads by Collection (Filtered: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
                    labels={'Collection': 'Collection', 'Downloads': 'Normalized Downloads'},
                    text='Normalized Downloads'
                )
                fig_dataset.update_xaxes(tickangle=45)
                fig_dataset.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                st.plotly_chart(fig_dataset, use_container_width=True)
        else:
            st.warning("Expected column 'Collection' not found in the data.")

        st.subheader("Raw Data for DICOM Downloads Over Time (GBytes)")

        # Drop the "File Size" column from dicom_df (was only necessary for normalized calculation)
        dicom_df = dicom_df.drop(columns=["File Size"])

        # Show data table
        st.dataframe(dicom_df, use_container_width=True)

        # Allow CSV export
        st.download_button(
            label="Download Data as CSV",
            data=dicom_df.to_csv(index=False).encode('utf-8'),
            file_name='downloads_dicom.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    create_app()
