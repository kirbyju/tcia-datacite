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
import concurrent.futures

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

def fetch_single_doi_stats(doi):
    """
    Fetches usage stats for a single DOI via the DataCite REST API (/dois/ endpoint).
    Returns a dictionary with views metrics.
    """
    # Using the specific DOI endpoint as requested
    url = f"https://api.datacite.org/dois/{doi}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            attributes = data.get('data', {}).get('attributes', {})
            
            # Extract viewsOverTime
            views_over_time = attributes.get('viewsOverTime', [])
            
            # Filter for months with > 0 views
            active_months_data = [item for item in views_over_time if item.get('total', 0) > 0]
            
            total_views = sum(item['total'] for item in active_months_data)
            active_months_count = len(active_months_data)
            
            return {
                'doi': doi,
                'API_ViewCount': total_views,
                'API_ActiveViewMonths': active_months_count
            }
            
    except Exception as e:
        pass
        
    return {
        'doi': doi, 
        'API_ViewCount': 0, 
        'API_ActiveViewMonths': 0
    }

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_datacite_stats_parallel(doi_list):
    """
    Fetches usage stats in parallel using ThreadPoolExecutor.
    """
    results = {}
    progress_text = "Fetching detailed usage stats from DataCite..."
    my_bar = st.progress(0, text=progress_text)
    
    # Using threads is efficient for I/O bound tasks like API requests
    # Cap workers to 10-15 to be polite to the API
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_doi = {executor.submit(fetch_single_doi_stats, doi): doi for doi in doi_list}
        
        completed_count = 0
        total_count = len(doi_list)
        
        for future in concurrent.futures.as_completed(future_to_doi):
            data = future.result()
            results[data['doi'].lower()] = data
            
            completed_count += 1
            if completed_count % 5 == 0 or completed_count == total_count:
                my_bar.progress(completed_count / total_count, text=f"{progress_text} ({completed_count}/{total_count})")
    
    my_bar.empty()
    return results

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
        
        st.info("""
        **Calculation Details:**
        *   **Monthly Page Views:** Calculated as `Total Views / Active Months`. This is retrieved directly from DataCite's API by counting months where views > 0.
        *   **Monthly Citations:** Calculated as `Total Citations / Months Since Creation`. This represents the dataset's long-term citation rate.
        """)

        # --- 1. Fetch & Calculate Metrics ---

        # A. Page Views (from REST API /dois/, parallelized)
        if 'DOI' in df.columns:
            doi_list = df['DOI'].dropna().unique().tolist()
            api_stats = fetch_datacite_stats_parallel(doi_list)
            
            # Map results
            df['API_ViewCount'] = df['DOI'].apply(lambda x: api_stats.get(x.lower(), {}).get('API_ViewCount', 0))
            df['API_ActiveViewMonths'] = df['DOI'].apply(lambda x: api_stats.get(x.lower(), {}).get('API_ActiveViewMonths', 0))
            
            # Filter: Drop rows with 0 page views
            df = df[df['API_ViewCount'] > 0]
            
            # Calculate Monthly Views (Active Months Only) & Round to Whole Number
            df['PageViewsMonthly'] = df.apply(
                lambda row: int(round(row['API_ViewCount'] / row['API_ActiveViewMonths'])) if row['API_ActiveViewMonths'] > 0 else 0, 
                axis=1
            )
        else:
            st.warning("DOI column missing. Cannot fetch detailed stats.")
            df['PageViewsMonthly'] = 0
            df['API_ActiveViewMonths'] = 0

        # B. Citations (Standard Age-based)
        df['Created'] = pd.to_datetime(df['Created']).dt.tz_localize(None)
        CURRENT_DATE = pd.Timestamp('now')
        
        # Months since creation (minimum 1 to avoid div/0)
        df['MonthsExistence'] = (CURRENT_DATE.year - df['Created'].dt.year) * 12 + (CURRENT_DATE.month - df['Created'].dt.month) + 1
        df['MonthsExistence'] = df['MonthsExistence'].apply(lambda x: max(x, 1))
        
        df['CitationsMonthly'] = (df['CitationCount'] / df['MonthsExistence']).round(2)

        # --- 2. Display Table ---
        column_order = [
            "DOI", "Identifier", "API_ViewCount", "PageViewsMonthly", "API_ActiveViewMonths",
            "CitationCount", "CitationsMonthly", "MonthsExistence",
            "ReferenceCount", "URL", "Title", "Related", "Created", "Updated", "Version", 
            "Rights", "RightsURI", "CreatorNames", "Description", "FundingReferences"
        ]
        available_cols = [col for col in column_order if col in df.columns]
        df_display = df[available_cols].copy()
        
        # Rename for clarity
        df_display = df_display.rename(columns={'API_ViewCount': 'TotalViews (API)'})
        
        if "TotalViews (API)" in df_display.columns:
            df_display = df_display.sort_values(by="TotalViews (API)", ascending=False)
        df_display = df_display.reset_index(drop=True)

        st.subheader("Explore Page Views and Citation Counts")
        st.dataframe(filter_dataframe(df_display))

        # --- 3. Treemaps ---
        col_treemap1, col_treemap2 = st.columns(2)

        with col_treemap1:
            # Treemap: Page Views
            fig_views = px.treemap(
                df,
                path=['Identifier'],
                values='PageViewsMonthly',
                title='Monthly Average Page Views (Active Months Only)',
                hover_name='Title',
                hover_data=['API_ViewCount', 'API_ActiveViewMonths', 'CitationCount']
            )
            fig_views.update_traces(
                hovertemplate="<b>%{hovertext}</b><br><br>" +
                              "Avg Views/Month: %{value}<br>" +
                              "Total Views: %{customdata[0]}<br>" +
                              "Active Months: %{customdata[1]}<br>" +
                              "Total Citations: %{customdata[2]}<extra></extra>"
            )
            st.plotly_chart(fig_views, use_container_width=True)

        with col_treemap2:
            # Treemap: Citations
            fig_citations = px.treemap(
                df,
                path=['Identifier'],
                values='CitationsMonthly',
                title='Monthly Average Citations (Lifetime)',
                hover_name='Title',
                hover_data=['CitationCount', 'MonthsExistence', 'API_ViewCount']
            )
            fig_citations.update_traces(
                hovertemplate="<b>%{hovertext}</b><br><br>" +
                              "Avg Citations/Month: %{value}<br>" +
                              "Total Citations: %{customdata[0]}<br>" +
                              "Dataset Age (Months): %{customdata[1]}<br>" +
                              "Total Views: %{customdata[2]}<extra></extra>"
            )
            st.plotly_chart(fig_citations, use_container_width=True)

        # --- 4. Scatter Plot ---
        # Using API_ViewCount for consistency with the new data source
        fig_scatter = px.scatter(
            df,
            x='API_ViewCount',
            y='CitationCount',
            hover_data=['Identifier', 'URL', 'Title'],
            title='Dataset Views vs Citations (Totals)',
            labels={'API_ViewCount': 'Total Views (API)', 'CitationCount': 'Total Citations'}
        )
        if len(df) > 1:
            m, b = np.polyfit(df['API_ViewCount'], df['CitationCount'], 1)
            fig_scatter.add_trace(
                go.Scatter(
                    x=df['API_ViewCount'],
                    y=m * df['API_ViewCount'] + b,
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