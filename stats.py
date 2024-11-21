import streamlit as st
import pandas as pd
import requests
from tcia_utils import datacite
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as ET

st.set_page_config(page_title="TCIA Data Usage Statistics", layout="wide")

@st.cache_data
def load_data():
    """Load TCIA dataset information using datacite"""
    df = datacite.getDoi()
    return df

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

# load endnote dataset
@st.cache_data
def load_endnote_data():
    url = "https://cancerimagingarchive.net/endnote/Pubs_basedon_TCIA.xml"
    response = requests.get(url)
    if response.status_code == 200:
        with open("Pubs_basedon_TCIA.xml", "wb") as f:
            f.write(response.content)

    endnote = parse_xml('Pubs_basedon_TCIA.xml')

    # remove any leading/trailing spaces
    endnote['electronic-resource-num'] = endnote['electronic-resource-num'].str.strip()
    return endnote

def create_app():
    # Sidebar for navigation and logo
    with st.sidebar:
        # Logo at the top of the sidebar
        st.image("https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-01.png", use_container_width=True)

        # Navigation selection
        page = st.radio("Select View", [
            "Dataset Citations by Year",
            "Citations and Page Views by Dataset",
            "Views vs Citations Scatter Plot"
        ])

    # Main content area
    st.title("TCIA Data Usage Statistics")
    st.write("Explore citation counts and page views for TCIA datasets.  Select from the available reports in the left sidebar.")

    # Load endnote pubs data
    pubs_df = load_endnote_data()

    # Load datacite data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    st.caption("Tip: Use the zoom and pan tools in the plot toolbar to explore specific regions")

    # Conditional rendering based on sidebar selection
    if page == "Dataset Citations by Year":
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
        fig.add_trace(go.Bar(x=pubs_per_year.index, y=pubs_per_year.values, name='Publications per Year'))
        fig.add_trace(go.Scatter(x=cumulative_pubs.index, y=cumulative_pubs.values, mode='lines+markers', name='Cumulative Publications', yaxis='y2'))
        fig.update_layout(
            title='Dataset Citations Over Time',
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

        # Interactive table with citation details
        st.subheader("Citation Details")

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

    elif page == "Views vs Citations Scatter Plot":
        # Scatter plot comparing views vs citations
        fig = px.scatter(
            df,
            x='ViewCount',
            y='CitationCount',
            hover_data=['Identifier', 'URL', 'Title'],
            title='Dataset Views vs Citations (Interactive)',
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

if __name__ == "__main__":
    create_app()
