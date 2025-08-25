import streamlit as st
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn import metrics

import urllib.parse
import urllib.request
import xml.dom.minidom
import time
import re
import requests
from geopy.distance import geodesic
import streamlit.components.v1 as components
import base64

# Set page config
st.set_page_config(page_title="BC Address Analyzer", layout="wide")

# Theme toggle switch (on/off)

# --- Detect Streamlit version ---
def has_toggle():
    major, minor, *_ = map(int, st.__version__.split("."))
    return major > 1 or (major == 1 and minor >= 25)

# --- Initialize theme_mode in session ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = False

# --- Logo ---
st.sidebar.image("bc.png", use_container_width=True)

# --- UI Toggle ---
st.sidebar.markdown("### 🌗 Theme Mode")

if has_toggle():
    # ✅ Native toggle switch (Streamlit ≥ 1.25)
    st.session_state.theme_mode = st.sidebar.toggle(
        "Dark Mode", value=st.session_state.theme_mode
    )
else:
    # ⛳ Fallback to button toggle
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        if st.button("⏻", key="dark_toggle_btn"):
            st.session_state.theme_mode = not st.session_state.theme_mode
    with col2:
        label = "🌑 Dark Mode: ON" if st.session_state.theme_mode else "☀️ Dark Mode: OFF"
        st.markdown(f"<div style='padding-top: 0.25em; font-weight: 600;'>{label}</div>", unsafe_allow_html=True)

# --- THEME CSS ---
if st.session_state.theme_mode:
    st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        background-color: #1E1E1E;
        color: #E0E0E0;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stButton > button {
        background-color: #333;
        color: white;
        font-weight: bold;
        border-radius: 0.4rem;
        padding: 0.4rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        background-color: #F0F2F6;
        color: #333;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        border-radius: 0.4rem;
        padding: 0.4rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Clean header

st.markdown("""
<h1 style="text-align: left; color: #2E86C1; font-weight: 700; margin-bottom: 0.2em;">Precariously Housed Patients Address Analysis</h1>
<p style="text-align: left; font-size: 1.1em; color: gray; margin-top: 0;">Upload, explore, analyze, visualize, and model patient address-based datasets from Vancouver Coastal Health.</p>
""", unsafe_allow_html=True)

# Tabs
st.markdown("---")
tabs = st.tabs(["&nbsp;&nbsp;📁 Upload&nbsp;&nbsp;", "&nbsp;&nbsp;🔎 Data Preprocessing&nbsp;&nbsp;", "&nbsp;&nbsp;📊 Descriptive&nbsp;&nbsp;", "&nbsp;&nbsp;📈 Visualization&nbsp;&nbsp;", "&nbsp;&nbsp;💻 Modeling&nbsp;&nbsp;"])
tables = {}
dataframes = {}
df = pd.DataFrame({}) 
df1 = pd.DataFrame({}) 
final_df = pd.DataFrame({})
flag = 0
# Session state init
if 'tables' not in st.session_state:
    st.session_state.tables = {}

if 'original_tables' not in st.session_state:
    st.session_state.original_tables = {}
    st.session_state.original_table_name_map = {}
    st.session_state.tables = {}
    st.session_state.table_name_map = {}



# Upload Tab
with tabs[0]:
    st.subheader("📁 Upload your Datasets")
    files = st.file_uploader("Drag and drop file(s)", type="csv", accept_multiple_files=True)
    if files:
        for i, file in enumerate(files):
            filename = file.name.replace(".csv", "")
            short_name = f"Table{i+1}"

            if short_name in st.session_state.original_tables:
                continue  # already uploaded

            display_name = f"{short_name} ({filename})"
            df = pd.read_csv(file)

            # Store original and also copy to working tables
            st.session_state.original_tables[short_name] = df
            st.session_state.original_table_name_map[short_name] = display_name

            st.session_state.tables[short_name] = df
            st.session_state.table_name_map[short_name] = display_name

    # Show original uploaded tables
    if st.session_state.original_tables:
        st.subheader("📂 Original Uploaded Tables")
        for short_name, df in st.session_state.original_tables.items():
            display_name = st.session_state.original_table_name_map[short_name]
            st.write(f"📋 **{display_name}**")
            st.dataframe(df.head())

    # Final output
    if st.session_state.tables:
        st.markdown("---")
        st.subheader("Results")
        keys = list(st.session_state.tables.keys())

        # Allow selecting multiple tables
        selected_tables = st.multiselect("Choose tables to display", keys, default=keys, key="selected_tables")

        if selected_tables:
            for table_name in selected_tables:
                st.info(f"✅ Table: {table_name}")
                st.dataframe(st.session_state.tables[table_name].head(10))
                st.markdown("---")
        
        col1, col2, _ = st.columns([1, 1, 1])

        with col1:
            if st.button("🔄 Reset All Tables"):
                for key in ["original_tables", "original_table_name_map", "tables", "table_name_map"]:
                    st.session_state.pop(key, None)
                st.rerun()

# EDA Tab
with tabs[1]:
    # --- Canada Post AddressComplete API functions ---
    def find_address(Key, SearchTerm):
        base_url = "http://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/xmla.ws?"
        params = {
            "Key": Key,
            "SearchTerm": SearchTerm,
            "Country": "CAN",
            "LanguagePreference": "EN",
            "MaxSuggestions": "1",
            "MaxResults": "1"
        }
        url = base_url + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()

        doc = xml.dom.minidom.parseString(xml_data)
        data_nodes = doc.getElementsByTagName("Row")
        if not data_nodes:
            return None

        id = data_nodes[0].getAttribute("Id")
        next_action = data_nodes[0].getAttribute("Next")

        # Loop if more steps are needed
        while next_action == "Find":
            params = {
                "Key": Key,
                "SearchTerm": "",
                "LastId": id,
                "Country": "CAN",
                "LanguagePreference": "EN",
                "MaxSuggestions": "1",
                "MaxResults": "1"
            }
            url = base_url + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()

            doc = xml.dom.minidom.parseString(xml_data)
            data_nodes = doc.getElementsByTagName("Row")
            if not data_nodes:
                return None

            id = data_nodes[0].getAttribute("Id")
            next_action = data_nodes[0].getAttribute("Next")

        return id

    def retrieve_full_address(Key, Id):
        base_url = "http://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Retrieve/v2.11/xmla.ws?"
        params = {
            "Key": Key,
            "Id": Id
        }
        url = base_url + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()
        doc = xml.dom.minidom.parseString(xml_data)
        schema_nodes = doc.getElementsByTagName("Column")
        data_nodes = doc.getElementsByTagName("Row")
        if not data_nodes:
            return None

        row = {
            col.getAttribute("Name"): data_nodes[0].getAttribute(col.getAttribute("Name"))
            for col in schema_nodes
        }
        return row

    def parse_address_fallback(full_address):
        result = {
        'E_SubBuilding': '',
        'E_BuildingNumber': '',
        'E_Street': '',
        'E_StreetType': '',
        'E_City': '',
        'E_ProvinceCode': 'BC',  # Always BC
        'E_PostalCode': '',
        'E_Country': ''
        }

        try:
            parts = [p.strip() for p in full_address.split(',') if p.strip()]
            if len(parts) >= 4:
                # Example: ["1666 W 75th Ave", "V6P 6G2", "Vancouver", "Canada"]
                street_part = parts[0]
                result['E_PostalCode'] = parts[1]
                result['E_City'] = parts[2]
                result['E_Country'] = parts[3]
            elif len(parts) == 3:
                street_part, result['E_PostalCode'], result['E_City'] = parts
                result['E_Country'] = "Canada"
            elif len(parts) == 2:
                street_part, result['E_PostalCode'] = parts
                result['E_Country'] = "Canada"
            else:
                street_part = parts[0]

            # Parse street line: e.g., "409 Granville St #256"
            match = re.match(
                r"(\d+)\s+([\w\s]+?)\s+(St|Ave|Street|Avenue|Blvd|Rd|Road|Dr|Drive|Way|Lane|Ln|Cres|Pl|Place|Court|Ct)\s*(#\d+)?",
                street_part.strip()
            )
            if match:
                result['E_BuildingNumber'] = match.group(1)
                result['E_Street'] = match.group(2).strip()
                result['E_StreetType'] = match.group(3)
                result['E_SubBuilding'] = match.group(4) if match.group(4) else ''

        except Exception as e:
            print(f"Fallback parsing failed: {e}")

        return result

    # --- Google API Key ---
    # API_KEY = ''  # Replace with your actual API key

    # # --- Function to get coordinates ---
    # def get_lat_lon(postal_code, country='Canada'):
    #     url = 'https://maps.googleapis.com/maps/api/geocode/json'
    #     params = {
    #         'address': f'{postal_code}, {country}',
    #         'key': API_KEY
    #     }

    #     try:
    #         response = requests.get(url, params=params)
    #         data = response.json()

    #         if data['status'] == 'OK':
    #             location = data['results'][0]['geometry']['location']
    #             return location['lat'], location['lng']
    #         else:
    #             print(f"[Google API] Error for {postal_code}: {data['status']}")
    #             return None, None
    #     except Exception as e:
    #         print(f"[Error] {postal_code}: {e}")
    #         return None, None
    
    # =================================================================
    # CENTRALIZED DATE UTILITY FUNCTIONS
    # =================================================================
    
    def calculate_days_between(date1, date2):
        """
        Calculate the number of days between two dates.
        
        This centralized function handles date difference calculations consistently
        across the application. It provides robust error handling and type conversion.
        
        Args:
            date1: First date (can be string, datetime, or pandas datetime)
            date2: Second date (can be string, datetime, or pandas datetime)
            
        Returns:
            int: Number of days between date2 and date1 (date2 - date1)
                 Positive if date2 is after date1, negative if before
                 None if either date is invalid/None
                 
        Examples:
            >>> calculate_days_between('2024-01-01', '2024-01-03')
            2
            >>> calculate_days_between('2024-01-03', '2024-01-01')
            -2
            >>> calculate_days_between(None, '2024-01-01')
            None
        """
        try:
            # Convert to pandas datetime if not already
            if pd.isna(date1) or pd.isna(date2):
                return None
                
            date1_dt = pd.to_datetime(date1)
            date2_dt = pd.to_datetime(date2)
            
            # Calculate difference in days
            diff = (date2_dt - date1_dt).days
            return int(diff)
            
        except Exception as e:
            print(f"Error calculating date difference: {e}")
            return None
        
    st.subheader("🔎 Data Preprocessing")
    
    # Initialize variables
    df = pd.DataFrame()
    selected_col = []
    selected_table = None
    flag = 0
    
    if st.checkbox("Choose the Address columns to elementize it."):
        # First, let user choose which table to work with
        if st.session_state.tables:
            table_keys = list(st.session_state.tables.keys())
            selected_table = st.selectbox(
                "Select table to work with:",
                table_keys,
                key="table_select_elementize"
            )
            df = st.session_state.tables[selected_table].copy()
            
            # Then let user select columns from the chosen table
            selected_col = st.multiselect(
                "Filter Columns:",
                df.columns,
                key="unique_select_vars"
            )
            st.text(" ")
        else:
            st.warning("⚠️ Please upload a dataset first.")
            selected_col = []

    if st.checkbox("Elementize the Addresses?"):
        if st.session_state.tables and 'selected_col' in locals() and selected_col:
            # Use the same table selected above
            table_keys = list(st.session_state.tables.keys())
            if 'selected_table' in locals():
                df = st.session_state.tables[selected_table].copy()
            else:
                # Fallback to first table if no table was selected
                df = st.session_state.tables[table_keys[0]].copy()
                
            df_tmp = df[selected_col]
            # Combine address lines into one
            df["fullAddress"] = df_tmp.fillna("").agg(lambda x: ", ".join([str(v) for v in x if str(v).strip()]), axis=1)
            
            api_key = "ZX12-HA39-BE19-ZZ84"
            elementized_data = []
            parsed = {}
            for _, row in df.iterrows():
                full_address = row["fullAddress"]
                # precision = row.get("PrecisionPoints", 0)
                
                # if precision >= 99:
                try:
                    addr_id = find_address(api_key, full_address)
                    if addr_id:
                        parsed_raw = retrieve_full_address(api_key, addr_id)
                        parsed = {
                            "E_SubBuilding": parsed_raw.get("SubBuilding", ""),
                            "E_BuildingNumber": parsed_raw.get("BuildingNumber", ""),
                            "E_Street": parsed_raw.get("Street", ""),
                            "E_StreetType": parsed_raw.get("StreetType", ""),
                            "E_City": parsed_raw.get("City", ""),
                            "E_ProvinceCode": parsed_raw.get("ProvinceCode", ""),
                            "E_PostalCode": parsed_raw.get("PostalCode", ""),
                            "E_Country": parsed_raw.get("CountryName", "")
                        }
                    else:
                        parsed = {}
                        parsed = parse_address_fallback(full_address)
                except Exception:
                    parsed = {}
                    parsed = parse_address_fallback(full_address)
                # else:
                #     # Fallback to basic parsing for low-precision data
                #     parsed = {}
                #     parsed = parse_address_fallback(full_address)
                elementized_data.append(parsed)
                #time.sleep(0.2)  # Avoid rate limits

            # --- Step 3: Merge parsed data and drop address lines ---
            address_df = pd.DataFrame(elementized_data)
            df.reset_index(drop=True, inplace=True)
            final_df = pd.concat([df, address_df], axis=1)
            final_df.drop(columns=["fullAddress"], errors="ignore", inplace=True)

            st.dataframe(final_df.head())
            
            # Allow user to create a new table with the elementized results
            st.write("**Save Elementized Results**")
            
            elementized_table_name = st.text_input(
                "New elementized table name:",
                value=f"Elementized_{selected_table}" if selected_table else "Elementized_Table",
                key="elementized_table_name"
            )
            
           
            if st.button("💾 Save Elementized Table", key="save_elementized_btn"):
                if elementized_table_name.strip():
                    # Store the elementized table in session state
                    st.session_state.tables[elementized_table_name] = final_df.copy()
                    st.session_state.table_name_map[elementized_table_name] = elementized_table_name
                    
                    st.success(f"✅ Elementized table saved as: **{elementized_table_name}**")
                    st.write(f"📊 **Table contains:** {len(final_df)} rows and {len(final_df.columns)} columns")
                else:
                    st.error("❌ Please enter a valid table name.")
            
            df1 = final_df.copy()
            flag = 1
        else:
            st.warning("⚠️ Please select a table and address columns first.")
            df1 = pd.DataFrame()  # Empty dataframe as fallback
            flag = 0

    if st.checkbox("Merge Discharge to emergency visit records?"):
        if st.session_state.tables and len(st.session_state.tables) >= 2:
            # Let user select tables for merging
            table_keys = list(st.session_state.tables.keys())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Emergency Visit Records Table**")
                visit_table = st.selectbox(
                    "Select visit records table:",
                    table_keys,
                    key="visit_table_select"
                )
                visit_df = st.session_state.tables[visit_table].copy()
                st.write(f"Columns: {list(visit_df.columns)}")
                
                # Select date column for visits
                visit_date_col = st.selectbox(
                    "Select visit date column:",
                    visit_df.columns,
                    key="visit_date_col"
                )
            
            with col2:
                st.write("**Discharge Records Table**")
                discharge_table = st.selectbox(
                    "Select discharge records table:",
                    [t for t in table_keys if t != visit_table],
                    key="discharge_table_select"
                )
                discharge_df = st.session_state.tables[discharge_table].copy()
                st.write(f"Columns: {list(discharge_df.columns)}")
                
                # Select date column for discharge
                discharge_date_col = st.selectbox(
                    "Select discharge date column:",
                    discharge_df.columns,
                    key="discharge_date_col"
                )
            
            st.write("### ⚙️ Merge Configuration")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Matching criteria
                st.write("**Matching Criteria**")
                match_client_id = st.checkbox("Match by ClientID", value=True, key="match_client_id")
                
                if match_client_id:
                    visit_client_col = st.selectbox(
                        "ClientID column in visit table:",
                        visit_df.columns,
                        key="visit_client_col"
                    )
                    discharge_client_col = st.selectbox(
                        "ClientID column in discharge table:",
                        discharge_df.columns,
                        key="discharge_client_col"
                    )
                
                # Date matching window
                date_tolerance = st.number_input(
                    "Date matching tolerance (days):",
                    min_value=0,
                    max_value=30,
                    value=1,
                    help="Discharge date should be within this many days after visit date",
                    key="date_tolerance"
                )
            
            with col4:
                st.write("**Output Configuration**")
                new_table_name = st.text_input(
                    "New merged table name:",
                    value="Merged_Visit_Discharge",
                    key="merged_table_name"
                )
                
                # Create expected merged column structure for output selection
                expected_merge_cols = []
                
                # Add visit columns
                for col in visit_df.columns:
                    expected_merge_cols.append(col)
                
                # Add discharge columns (with prefixes for conflicts)
                for col in discharge_df.columns:
                    if col in visit_df.columns and col != discharge_date_col:
                        expected_merge_cols.append(f"Discharge_{col}")
                    else:
                        expected_merge_cols.append(col)
                
                # Add the calculated column
                expected_merge_cols.append('Days_Between_Visit')
                
                # Remove duplicates while preserving order
                unique_merge_cols = []
                for col in expected_merge_cols:
                    if col not in unique_merge_cols:
                        unique_merge_cols.append(col)
                
                # Column selection for final output
                st.write("**Select columns to include in final merged table:**")
                output_cols_selected = st.multiselect(
                    "Choose columns for the merged output:",
                    unique_merge_cols,
                    default=unique_merge_cols,  # Select all columns by default
                    key="output_cols_select"
                )
            
            # Merge button and logic
            if st.button("🔗 Merge Tables", key="merge_tables_btn"):
                try:
                    # Use ALL columns for internal merge processing
                    visit_work = visit_df.copy()
                    discharge_work = discharge_df.copy()
                    
                    # Convert date columns to datetime
                    visit_work[visit_date_col] = pd.to_datetime(visit_work[visit_date_col])
                    discharge_work[discharge_date_col] = pd.to_datetime(discharge_work[discharge_date_col])
                    
                    # Create a list to store merged results
                    merged_results = []
                    
                    # Perform the merge logic
                    for _, visit_row in visit_work.iterrows():
                        visit_date = visit_row[visit_date_col]
                        
                        if match_client_id:
                            client_id = visit_row[visit_client_col]
                            # Find discharge records for the same client
                            candidate_discharges = discharge_work[
                                discharge_work[discharge_client_col] == client_id
                            ].copy()
                        else:
                            candidate_discharges = discharge_work.copy()
                        
                        # Find discharges within the date tolerance
                        candidate_discharges['date_diff'] = candidate_discharges[discharge_date_col].apply(
                            lambda discharge_date: calculate_days_between(visit_date, discharge_date)
                        )
                        
                        # Filter for discharges that occur after visit within tolerance
                        valid_discharges = candidate_discharges[
                            (candidate_discharges['date_diff'].notna()) &
                            (candidate_discharges['date_diff'] >= 0) & 
                            (candidate_discharges['date_diff'] <= date_tolerance)
                        ]
                        
                        if not valid_discharges.empty:
                            # Take the closest discharge (minimum date difference)
                            closest_discharge = valid_discharges.loc[
                                valid_discharges['date_diff'].idxmin()
                            ]
                            
                            # Merge the records
                            merged_row = visit_row.copy()
                            
                            # Add discharge columns (with prefix to avoid conflicts)
                            for col in discharge_df.columns:
                                if col in visit_df.columns and col != discharge_date_col:
                                    merged_row[f"Discharge_{col}"] = closest_discharge[col]
                                else:
                                    merged_row[col] = closest_discharge[col]
                            
                            # Add date difference as additional info
                            merged_row['Days_Between_Visit'] = closest_discharge['date_diff']
                            
                        else:
                            # No matching discharge found
                            merged_row = visit_row.copy()
                            
                            # Add empty discharge columns
                            for col in discharge_df.columns:
                                if col in visit_df.columns and col != discharge_date_col:
                                    merged_row[f"Discharge_{col}"] = None
                                else:
                                    merged_row[col] = None
                            
                            merged_row['Days_Between_Visit'] = None
                        
                        merged_results.append(merged_row)
                    
                    # Create the initial merged dataframe with all columns
                    merged_df_full = pd.DataFrame(merged_results)
                    
                    # Remove the temporary date_diff column if it exists
                    merged_df_full.drop(columns=['date_diff'], errors='ignore', inplace=True)
                    
                    # Apply user's column selection for final output
                    if output_cols_selected:
                        # Filter to only include selected columns that actually exist
                        available_cols = [col for col in output_cols_selected if col in merged_df_full.columns]
                        merged_df = merged_df_full[available_cols].copy()
                    else:
                        # If no columns selected, use all columns
                        merged_df = merged_df_full.copy()
                    
                    # Store the merged table
                    st.session_state.tables[new_table_name] = merged_df
                    st.session_state.table_name_map[new_table_name] = new_table_name
                    
                    st.success(f"✅ Successfully merged tables! Created: **{new_table_name}**")
                    st.write(f"📊 **Merge Statistics:**")
                    
                    total_visits = len(visit_work)
                    matched_records = merged_df_full['Days_Between_Visit'].notna().sum()
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("Total Visits", total_visits)
                    with col_stats2:
                        st.metric("Matched Records", matched_records)
                    with col_stats3:
                        match_rate = (matched_records / total_visits * 100) if total_visits > 0 else 0
                        st.metric("Match Rate", f"{match_rate:.1f}%")
                    
                    # Display merged data preview
                    st.write("📋 **Merged Data Preview:**")
                    st.dataframe(merged_df.head(100))
                    
                except Exception as e:
                    st.error(f"❌ Error during merge: {str(e)}")
                    st.write("Please check your date columns and ensure they contain valid dates.")
        else:
            st.warning("⚠️ You need at least 2 tables uploaded to perform a merge operation.")
    
    # COMMENTED OUT - Get coordinate from an address postalcode function
    # if st.checkbox("Get coordinate from an address postalcode? (via Google API)"):
    #     if flag == 1 and 'final_df' in locals():
    #         temp_df = final_df.copy()
    #     elif selected_table and not df.empty:
    #         temp_df = df.copy()
    #     else:
    #         st.warning("⚠️ Please select a table first or elementize addresses.")
    #         temp_df = pd.DataFrame()

    #     if not temp_df.empty:
    #         # --- Ensure postalCode exists ---
    #         if 'PostalCode' not in temp_df.columns:
    #             st.error("❌ PostalCode column not found in dataset.")
    #         else:
    #             # --- Get unique postal codes ---
    #             unique_postals = temp_df['PostalCode'].dropna().astype(str).unique()
    #             # --- Map postal codes to coordinates ---
    #             postal_to_coords = {}
    #             for pc in unique_postals:
    #                 if pc.strip():
    #                     lat, lon = get_lat_lon(pc)
    #                     postal_to_coords[pc] = (lat, lon)
    #                     time.sleep(0.2)  # Rate limiting

    #             # --- Append lat/lon to temp_df ---
    #             temp_df['Latitude'] = temp_df['PostalCode'].astype(str).map(lambda x: postal_to_coords.get(x, (None, None))[0])
    #             temp_df['Longitude'] = temp_df['PostalCode'].astype(str).map(lambda x: postal_to_coords.get(x, (None, None))[1])
    #             st.dataframe(temp_df.head())
                
    #             # Update final_df with coordinates
    #             final_df = temp_df.copy()
    #             st.session_state.final_df = final_df.copy()
    #     else:
    #         # If coordinate generation is checked but no valid data
    #         if selected_table and not df.empty:
    #             st.session_state.final_df = df.copy()
    #         else:
    #             st.session_state.final_df = pd.DataFrame()
    # else:
    #     # If coordinate generation is not checked, use the selected table or elementized data
    #     if flag == 1 and 'final_df' in locals():
    #         st.session_state.final_df = final_df.copy()
    #     elif selected_table and not df.empty:
    #         st.session_state.final_df = df.copy()
    #     else:
    #         st.session_state.final_df = pd.DataFrame()

    # Fallback for final_df when coordinate generation is commented out
    if flag == 1 and 'final_df' in locals():
        st.session_state.final_df = final_df.copy()
    elif selected_table and not df.empty:
        st.session_state.final_df = df.copy()
    else:
        st.session_state.final_df = pd.DataFrame()

# Descriptive Tab
with tabs[2]:
    st.subheader("📊 Descriptive Analysis (Address Based)")
    
    # Table selection for analysis
    if st.session_state.tables:
        st.write("**Select table for descriptive analysis:**")
        analysis_table = st.selectbox(
            "Choose table to analyze:",
            list(st.session_state.tables.keys()),
            key="descriptive_analysis_table"
        )
        
        if analysis_table:
            temp_df = st.session_state.tables[analysis_table].copy()
            st.write(f"📋 **Analyzing table: {analysis_table}**")
            st.dataframe(temp_df.head())
        else:
            st.warning("⚠️ Please select a table for analysis.")
            temp_df = pd.DataFrame()
    else:
        st.warning("⚠️ No tables available for analysis. Please upload and process data first.")
        temp_df = pd.DataFrame()
    
    # Only show analysis options if we have a valid dataframe
    if not temp_df.empty:
        if st.checkbox("⏱️ Move Frequency Relative to Patient"): 
            # Check if MoveFrequency column already exists and drop it to avoid conflicts
            if 'MoveFrequency' in temp_df.columns:
                temp_df = temp_df.drop(columns=['MoveFrequency'])
            
            move_freq = temp_df.groupby("ClientID").size().rename("MoveFrequency")
            temp_df = temp_df.merge(move_freq, on="ClientID", how="left")
            st.dataframe(temp_df)
            
            # Save results back to original table
            st.session_state.tables[analysis_table] = temp_df.copy()
            st.success(f"✅ Move Frequency analysis results saved to table: **{analysis_table}**")

        if st.checkbox("⚠️ Frequency of Missing or Incomplete Address Fields"):
            # Check if MissingFrequency column already exists and drop it to avoid conflicts
            if 'MissingFrequency' in temp_df.columns:
                temp_df = temp_df.drop(columns=['MissingFrequency'])
            
            incomplete_fields = ['E_Street', 'E_StreetType', 'E_PostalCode', 'E_City']
            
            temp_df["row_has_missing"] = temp_df[incomplete_fields].isnull().any(axis=1)
            temp_df["MissingFrequency"] = temp_df.groupby("ClientID")["row_has_missing"].transform("sum")
            temp_df.drop(columns="row_has_missing", inplace=True)
            st.dataframe(temp_df)
            
            # Save results back to original table
            st.session_state.tables[analysis_table] = temp_df.copy()
            st.success(f"✅ Missing Address Fields analysis results saved to table: **{analysis_table}**")

        if st.checkbox("🏢 Classification: % Residential vs Addresses"):
            # Check if Commercial column already exists and drop it to avoid conflicts
            if 'Commercial' in temp_df.columns:
                temp_df = temp_df.drop(columns=['Commercial'])
            
            def classify_address(street):
                if pd.isna(street):
                    return "Unknown"
                commercial_keywords = ["Office", "Centre", "Plaza", "Mall", "Business", "Tower"]
                if any(word.lower() in street.lower() for word in commercial_keywords):
                    return "Commercial"
                return "Residential"

            temp_df["address_class"] = temp_df["E_Street"].apply(classify_address)
            
            address_by_patient = (
                temp_df.groupby(["ClientID", "address_class"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["Residential", "Commercial"], fill_value=0)  # <- This ensures both columns exist
                .reset_index()
            )
            address_by_patient["Total"] = address_by_patient[["Residential", "Commercial"]].sum(axis=1)
            #address_by_patient["Residential"] = (address_by_patient["Residential"] / address_by_patient["Total"] * 100).round(1)
            address_by_patient["CommercialAddress"] = (address_by_patient["Commercial"] / address_by_patient["Total"] * 100).round(1)


            temp_df = temp_df.merge(
                address_by_patient[["ClientID", "Commercial"]],
                on="ClientID",
                how="left"
            )

            temp_df.drop(columns="address_class", inplace=True)

            st.dataframe(temp_df)
            
            # Save results back to original table
            st.session_state.tables[analysis_table] = temp_df.copy()
            st.success(f"✅ Residential vs Commercial classification results saved to table: **{analysis_table}**")

        if st.checkbox("🏥 Check ED Visit Centers Postal Code match with ED facilities"):
            # Table selection for ED facility matching
            if st.session_state.tables:
                st.write("**Select table for ED facility matching analysis:**")
                facility_analysis_table = st.selectbox(
                    "Choose table to analyze:",
                    list(st.session_state.tables.keys()),
                    key="facility_analysis_table"
                )
                
                if facility_analysis_table:
                    temp_df_facility = st.session_state.tables[facility_analysis_table].copy()
                    st.write(f"📋 **Analyzing table: {facility_analysis_table}**")
                    
                    # Check if FacilityMatched column already exists and drop it to avoid conflicts
                    if 'FacilityMatched' in temp_df_facility.columns:
                        temp_df_facility = temp_df_facility.drop(columns=['FacilityMatched'])
                    
                    st.write("**Column Selection:**")
                    
                    # Column selection for postal code to check
                    postal_code_columns = [col for col in temp_df_facility.columns if 'postal' in col.lower() or 'code' in col.lower()]
                    if not postal_code_columns:
                        postal_code_columns = list(temp_df_facility.columns)
                    
                    selected_postal_col = st.selectbox(
                        "Select column containing postal codes to check:",
                        postal_code_columns,
                        key="postal_code_col_select"
                    )
                    
                    st.write("**Upload ED Facility Files:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**ED Facilities File**")
                        primary_facilities_file = st.file_uploader(
                            "Upload CSV with ED facility postal codes:",
                            type="csv",
                            key="primary_facilities_upload"
                        )
                        
                    with col2:
                        st.write("**Non-Unique ED Facilities File**")
                        secondary_facilities_file = st.file_uploader(
                            "Upload CSV with non-unique ED facility postal codes:",
                            type="csv",
                            key="secondary_facilities_upload"
                        )
                    
                    # Check facility matches
                    def check_facility_match(postal_code_at_visit, primary_facilities_df, secondary_facilities_df):
                        """
                        Check if a postal code matches ED facility postal codes.
                        
                        Args:
                            postal_code_at_visit: Postal code to check
                            primary_facilities_df: DataFrame with ED facility postal codes
                            secondary_facilities_df: DataFrame with non-unique ED facility postal codes
                            
                        Returns:
                            str: "Yes" if match found, "No" if no match
                        """
                        if pd.isna(postal_code_at_visit):
                            return "No"
                        
                        # Clean and normalize postal code
                        postal_code_clean = str(postal_code_at_visit).strip().upper().replace(" ", "")
                        
                        # Check primary facilities
                        if primary_facilities_df is not None and not primary_facilities_df.empty:
                            for col in primary_facilities_df.columns:
                                if 'postal' in col.lower() or 'code' in col.lower():
                                    primary_codes = primary_facilities_df[col].dropna().astype(str).str.strip().str.upper().str.replace(" ", "")
                                    if postal_code_clean in primary_codes.values:
                                        return "Yes"
                        
                        # Check secondary facilities
                        if secondary_facilities_df is not None and not secondary_facilities_df.empty:
                            for col in secondary_facilities_df.columns:
                                if 'postal' in col.lower() or 'code' in col.lower():
                                    secondary_codes = secondary_facilities_df[col].dropna().astype(str).str.strip().str.upper().str.replace(" ", "")
                                    if postal_code_clean in secondary_codes.values:
                                        return "Yes"
                        
                        return "No"
                    
                    # Process facility matching if files are uploaded
                    if primary_facilities_file is not None or secondary_facilities_file is not None:
                        try:
                            # Load facility files
                            primary_df = None
                            secondary_df = None
                            
                            if primary_facilities_file is not None:
                                primary_df = pd.read_csv(primary_facilities_file)
                                st.write("**Primary Facilities Data Preview:**")
                                st.dataframe(primary_df.head())
                            
                            if secondary_facilities_file is not None:
                                secondary_df = pd.read_csv(secondary_facilities_file)
                                st.write("**Secondary Facilities Data Preview:**")
                                st.dataframe(secondary_df.head())
                            
                            if st.button("🔍 Check Facility Matches", key="check_facility_btn"):
                                # Apply facility matching function
                                temp_df_facility['FacilityMatched'] = temp_df_facility[selected_postal_col].apply(
                                    lambda x: check_facility_match(x, primary_df, secondary_df)
                                )
                                
                                # Display results
                                st.write("**Facility Matching Results:**")
                                match_summary = temp_df_facility['FacilityMatched'].value_counts()
                                
                                col_stats1, col_stats2, col_stats3 = st.columns(3)
                                with col_stats1:
                                    total_records = len(temp_df_facility)
                                    st.metric("Total Records", total_records)
                                
                                with col_stats2:
                                    matched_count = match_summary.get("Yes", 0)
                                    st.metric("Facility Matches", matched_count)
                                
                                with col_stats3:
                                    match_rate = (matched_count / total_records * 100) if total_records > 0 else 0
                                    st.metric("Match Rate", f"{match_rate:.1f}%")
                                
                                st.dataframe(temp_df_facility)
                                
                                # Save results back to original table
                                st.session_state.tables[facility_analysis_table] = temp_df_facility.copy()
                                st.success(f"✅ ED Facility matching results saved to table: **{facility_analysis_table}**")
                        
                        except Exception as e:
                            st.error(f"❌ Error processing facility files: {str(e)}")
                            st.write("Please ensure your CSV files contain postal code columns.")
                    else:
                        st.info("📋 Please upload at least one ED facility file to proceed with matching.")
                else:
                    st.warning("⚠️ Please select a table for ED facility analysis.")
            else:
                st.warning("⚠️ No tables available for analysis. Please upload and process data first.")

        # COMMENTED OUT - Distance Moved Between Address Changes function
        # if st.checkbox("📍 Distance Moved Between Address Changes"):
        #     # Check if required columns exist
        #     if 'Latitude' not in temp_df.columns or 'Longitude' not in temp_df.columns:
        #         st.error("❌ Latitude and Longitude columns are required for distance calculation. Please ensure coordinates are generated first.")
        #     else:
        #         # Let user select the date column for chronological sorting
        #         potential_date_cols = []
        #         for col in temp_df.columns:
        #             # Check if column might be a date column
        #             if any(keyword in col.lower() for keyword in ['date', 'time', 'start', 'end', 'created', 'updated']):
        #                 potential_date_cols.append(col)
        #             # Also check data type
        #             elif pd.api.types.is_datetime64_any_dtype(temp_df[col]):
        #                 if col not in potential_date_cols:
        #                     potential_date_cols.append(col)
        #         
        #         if not potential_date_cols:
        #             st.error("❌ **Date column is required for analysis!**")
        #             st.warning("⚠️ No date columns found for chronological sorting. Distance calculation requires a date column to order address changes chronologically.")
        #             st.info("💡 **Tip:** Make sure your dataset contains columns with 'date', 'time', 'start', 'end', 'created', or 'updated' in their names, or columns with datetime data types.")
        #         else:
        #             st.write("**Date Column Selection:**")
        #             selected_date_col = st.selectbox(
        #                 "Choose date column for chronological ordering of addresses:", 
        #                 potential_date_cols,
        #                 key="distance_date_col"
        #             )
        #             
        #             # Convert to datetime if needed
        #             if not pd.api.types.is_datetime64_any_dtype(temp_df[selected_date_col]):
        #                 try:
        #                     temp_df[selected_date_col] = pd.to_datetime(temp_df[selected_date_col])
        #                 except:
        #                     st.error(f"❌ Unable to convert {selected_date_col} to datetime format.")
        #                     selected_date_col = None
        #             
        #             if selected_date_col:
        #                 def compute_move_distances(group):
        #                     group = group.sort_values(selected_date_col)
        #                     coords = list(zip(group["Latitude"], group["Longitude"]))
        #                     distances = [0]
        #                     for i in range(1, len(coords)):
        #                         if None not in coords[i] and None not in coords[i - 1]:
        #                             distances.append(geodesic(coords[i - 1], coords[i]).km)
        #                         else:
        #                             distances.append(None)
        #                     return pd.Series(distances, index=group.index)

        #                 temp_df["MoveDistance(KM)"] = temp_df.groupby("ClientID").apply(compute_move_distances).reset_index(level=0, drop=True).round(1)
        #                 
        #                 # Show summary statistics
        #                 st.write("**Distance Movement Summary:**")
        #                 col1, col2, col3 = st.columns(3)
        #                 
        #                 with col1:
        #                     total_moves = (temp_df['MoveDistance(KM)'] > 0).sum()
        #                     st.metric("Total Address Changes", total_moves)
        #                 
        #                 with col2:
        #                     avg_distance = temp_df[temp_df['MoveDistance(KM)'] > 0]['MoveDistance(KM)'].mean()
        #                     st.metric("Average Distance (KM)", f"{avg_distance:.2f}" if not pd.isna(avg_distance) else "N/A")
        #                 
        #                 with col3:
        #                     max_distance = temp_df['MoveDistance(KM)'].max()
        #                     st.metric("Max Distance (KM)", f"{max_distance:.2f}" if not pd.isna(max_distance) else "N/A")
        #                 
        #                 st.dataframe(temp_df)

    st.subheader("📊 Descriptive Analysis (Patient Based)")
    
    if st.checkbox("🏠 Number of Patients Sharing Same Address"):
        # Table selection for this analysis
        if st.session_state.tables:
            st.write("**Select table for address sharing analysis:**")
            address_sharing_table = st.selectbox(
                "Choose table to analyze:",
                list(st.session_state.tables.keys()),
                key="address_sharing_table"
            )
            
            if address_sharing_table:
                temp_df_address = st.session_state.tables[address_sharing_table].copy()
                st.write(f"📋 **Analyzing table: {address_sharing_table}**")
                
                # Check if PatientsAtAddress column already exists and drop it to avoid conflicts
                if 'PatientsAtAddress' in temp_df_address.columns:
                    temp_df_address = temp_df_address.drop(columns=['PatientsAtAddress'])
                
                # Create a full address string for grouping
                address_columns = ['E_BuildingNumber', 'E_Street', 'E_StreetType', 'E_PostalCode', 'E_City']
                
                # Check if required columns exist
                missing_cols = [col for col in address_columns if col not in temp_df_address.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns for address analysis: {missing_cols}")
                    st.info("💡 **Tip:** This analysis requires columns named: E_BuildingNumber, E_Street, E_StreetType, E_PostalCode, E_City")
                else:
                    # Handle missing values and create a complete address string
                    temp_df_address['FullAddressKey'] = temp_df_address[address_columns].fillna('').apply(
                        lambda x: ', '.join([str(val).strip() for val in x if str(val).strip()]), axis=1
                    )
                    
                    # Count unique patients per address
                    address_patient_count = (
                        temp_df_address.groupby('FullAddressKey')['ClientID']
                        .nunique()
                        .rename('PatientsAtAddress')
                        .reset_index()
                    )
                    
                    # Merge back to main dataframe
                    temp_df_address = temp_df_address.merge(
                        address_patient_count,
                        on='FullAddressKey',
                        how='left'
                    )
                    
                    # Drop the temporary address key column
                    temp_df_address.drop(columns='FullAddressKey', inplace=True)
                    
                    # Display summary statistics
                    st.write("**Address Sharing Summary:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        unique_addresses = len(address_patient_count)
                        st.metric("Total Unique Addresses", unique_addresses)
                    
                    with col2:
                        max_patients = temp_df_address['PatientsAtAddress'].max()
                        st.metric("Max Patients per Address", max_patients)
                    
                    with col3:
                        shared_addresses = (temp_df_address['PatientsAtAddress'] > 1).sum()
                        st.metric("Shared Address Records", shared_addresses)
                    
                    st.dataframe(temp_df_address)
                    
                    # Save results back to original table
                    st.session_state.tables[address_sharing_table] = temp_df_address.copy()
                    st.success(f"✅ Patients Sharing Address analysis results saved to table: **{address_sharing_table}**")
            else:
                st.warning("⚠️ Please select a table for analysis.")
        else:
            st.warning("⚠️ No tables available for analysis. Please upload and process data first.")

# Visualization Tab
with tabs[3]:
    st.subheader("📈 Data Visualization")
    
    # Table selection for visualization
    if st.session_state.tables:
        st.write("**Select table for visualization:**")
        viz_table = st.selectbox(
            "Choose table to visualize:",
            list(st.session_state.tables.keys()),
            key="visualization_table"
        )
        
        if viz_table:
            viz_df = st.session_state.tables[viz_table].copy()
            st.write(f"📋 **Visualizing table: {viz_table}**")
            st.write(f"📊 **Table shape:** {viz_df.shape[0]} rows × {viz_df.shape[1]} columns")
            
            # Column selection for visualization
            st.write("**Select columns for visualization:**")
            viz_columns = st.multiselect(
                "Choose columns to include in visualization:",
                viz_df.columns.tolist(),
                key="viz_columns_select"
            )
            
            if viz_columns:
                # Filter dataframe to selected columns
                viz_data = viz_df[viz_columns].copy()
                
                # Display data preview
                with st.expander("📋 Data Preview", expanded=False):
                    st.dataframe(viz_data.head(10))
                    
                    # Basic statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Data Types:**")
                        st.write(viz_data.dtypes)
                    with col2:
                        st.write("**Missing Values:**")
                        st.write(viz_data.isnull().sum())
                
                # Plot type selection
                st.write("### 📊 Choose Visualization Type")
                
                plot_type = st.selectbox(
                    "Select plot type:",
                    [
                        "📊 Bar Chart",
                        "📈 Line Chart", 
                        "🔵 Scatter Plot",
                        "📦 Box Plot",
                        "📊 Histogram",
                        "🥧 Pie Chart",
                        "🔥 Heatmap",
                        "📊 Count Plot",
                        "📈 Distribution Plot",
                        "🎯 Pair Plot"
                    ],
                    key="plot_type_select"
                )
                
                # Visualization generation based on plot type
                try:
                    if plot_type == "📊 Bar Chart":
                        if len(viz_columns) >= 1:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("Select X-axis column:", viz_columns, key="bar_x")
                            with col2:
                                y_col = st.selectbox("Select Y-axis column (optional):", [None] + viz_columns, key="bar_y")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if y_col and y_col != x_col:
                                # Grouped bar chart
                                viz_data.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax)
                                ax.set_ylabel(y_col)
                            else:
                                # Count bar chart
                                viz_data[x_col].value_counts().head(20).plot(kind='bar', ax=ax)
                                ax.set_ylabel('Count')
                            
                            ax.set_title(f'Bar Chart: {x_col}')
                            ax.set_xlabel(x_col)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    elif plot_type == "📈 Line Chart":
                        if len(viz_columns) >= 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("Select X-axis column:", viz_columns, key="line_x")
                            with col2:
                                y_col = st.selectbox("Select Y-axis column:", [col for col in viz_columns if col != x_col], key="line_y")
                            
                            # Sort by x column for better line plot
                            sorted_data = viz_data.sort_values(x_col)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(sorted_data[x_col], sorted_data[y_col], marker='o')
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            ax.set_title(f'Line Chart: {y_col} vs {x_col}')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Line chart requires at least 2 columns.")
                    
                    elif plot_type == "🔵 Scatter Plot":
                        if len(viz_columns) >= 2:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                x_col = st.selectbox("Select X-axis column:", viz_columns, key="scatter_x")
                            with col2:
                                y_col = st.selectbox("Select Y-axis column:", [col for col in viz_columns if col != x_col], key="scatter_y")
                            with col3:
                                color_col = st.selectbox("Color by (optional):", [None] + [col for col in viz_columns if col not in [x_col, y_col]], key="scatter_color")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if color_col:
                                scatter = ax.scatter(viz_data[x_col], viz_data[y_col], c=pd.Categorical(viz_data[color_col]).codes, alpha=0.6)
                                plt.colorbar(scatter)
                            else:
                                ax.scatter(viz_data[x_col], viz_data[y_col], alpha=0.6)
                            
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            ax.set_title(f'Scatter Plot: {y_col} vs {x_col}')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Scatter plot requires at least 2 columns.")
                    
                    elif plot_type == "📦 Box Plot":
                        numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            selected_numeric = st.multiselect("Select numeric columns for box plot:", numeric_cols, default=numeric_cols[:3], key="box_cols")
                            if selected_numeric:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                viz_data[selected_numeric].boxplot(ax=ax)
                                ax.set_title('Box Plot')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.warning("⚠️ No numeric columns available for box plot.")
                    
                    elif plot_type == "📊 Histogram":
                        numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            col_to_plot = st.selectbox("Select column for histogram:", numeric_cols, key="hist_col")
                            bins = st.slider("Number of bins:", min_value=5, max_value=50, value=20, key="hist_bins")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(viz_data[col_to_plot].dropna(), bins=bins, alpha=0.7, edgecolor='black')
                            ax.set_xlabel(col_to_plot)
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Histogram: {col_to_plot}')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ No numeric columns available for histogram.")
                    
                    elif plot_type == "🥧 Pie Chart":
                        categorical_cols = viz_data.select_dtypes(include=['object', 'category']).columns.tolist()
                        if categorical_cols:
                            col_to_plot = st.selectbox("Select column for pie chart:", categorical_cols, key="pie_col")
                            top_n = st.slider("Show top N categories:", min_value=3, max_value=15, value=8, key="pie_top_n")
                            
                            value_counts = viz_data[col_to_plot].value_counts().head(top_n)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                            ax.set_title(f'Pie Chart: {col_to_plot}')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ No categorical columns available for pie chart.")
                    
                    elif plot_type == "🔥 Heatmap":
                        numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) >= 2:
                            correlation_matrix = viz_data[numeric_cols].corr()
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                            ax.set_title('Correlation Heatmap')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Heatmap requires at least 2 numeric columns.")
                    
                    elif plot_type == "📊 Count Plot":
                        categorical_cols = viz_data.select_dtypes(include=['object', 'category']).columns.tolist()
                        if categorical_cols:
                            col_to_plot = st.selectbox("Select column for count plot:", categorical_cols, key="count_col")
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            value_counts = viz_data[col_to_plot].value_counts().head(20)
                            ax.bar(range(len(value_counts)), value_counts.values)
                            ax.set_xticks(range(len(value_counts)))
                            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                            ax.set_xlabel(col_to_plot)
                            ax.set_ylabel('Count')
                            ax.set_title(f'Count Plot: {col_to_plot}')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ No categorical columns available for count plot.")
                    
                    elif plot_type == "📈 Distribution Plot":
                        numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            col_to_plot = st.selectbox("Select column for distribution plot:", numeric_cols, key="dist_col")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            viz_data[col_to_plot].dropna().hist(bins=30, alpha=0.7, density=True, ax=ax)
                            viz_data[col_to_plot].dropna().plot.density(ax=ax, color='red', linewidth=2)
                            ax.set_xlabel(col_to_plot)
                            ax.set_ylabel('Density')
                            ax.set_title(f'Distribution Plot: {col_to_plot}')
                            ax.legend(['Density', 'Histogram'])
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ No numeric columns available for distribution plot.")
                    
                    elif plot_type == "🎯 Pair Plot":
                        numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) >= 2:
                            if len(numeric_cols) > 5:
                                st.warning("⚠️ Too many numeric columns. Selecting first 5 for pair plot.")
                                selected_cols = numeric_cols[:5]
                            else:
                                selected_cols = numeric_cols
                            
                            # Create pair plot manually
                            n_cols = len(selected_cols)
                            fig, axes = plt.subplots(n_cols, n_cols, figsize=(12, 12))
                            
                            for i, col1 in enumerate(selected_cols):
                                for j, col2 in enumerate(selected_cols):
                                    ax = axes[i, j]
                                    if i == j:
                                        # Diagonal: histogram
                                        ax.hist(viz_data[col1].dropna(), bins=20, alpha=0.7)
                                        ax.set_title(col1)
                                    else:
                                        # Off-diagonal: scatter plot
                                        ax.scatter(viz_data[col2], viz_data[col1], alpha=0.5)
                                    
                                    if i == n_cols - 1:
                                        ax.set_xlabel(col2)
                                    if j == 0:
                                        ax.set_ylabel(col1)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Pair plot requires at least 2 numeric columns.")
                
                except Exception as e:
                    st.error(f"❌ Error creating visualization: {str(e)}")
                    st.write("Please check your data and column selections.")
                
            else:
                st.info("📋 Please select at least one column to visualize.")
        else:
            st.warning("⚠️ Please select a table for visualization.")
    else:
        st.warning("⚠️ No tables available for visualization. Please upload and process data first.")
    

# Modeling Tab
with tabs[4]:
    st.subheader("💻 Machine Learning Models")
    

# Final fallback warning
if __name__ == '__main__':
    if not st.session_state.tables:
        st.warning("⚠️ Please upload a dataset to begin.")