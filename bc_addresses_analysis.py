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
if 'join_round' not in st.session_state:
    st.session_state.join_round = 1
if 'keep_joining' not in st.session_state:
    st.session_state.keep_joining = False

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

    # Ask to join
    if len(st.session_state.tables) >= 2 and not st.session_state.keep_joining:
        choice = st.radio("Would you like to join tables?", ["No", "Yes"], key="initial_join_radio")
        if choice == "Yes":
            st.session_state.keep_joining = True

    # Join loop
    while st.session_state.keep_joining and len(st.session_state.tables) >= 2:
        round_id = st.session_state.join_round
        tables = st.session_state.tables
        table_names = list(tables.keys())

        st.subheader(f"Join Round {round_id}")

        display_map = st.session_state.table_name_map
        short_names = list(st.session_state.tables.keys())
        
        table_a_disp = st.selectbox("Select Table A", [display_map[k] for k in short_names], key=f"table_a_{round_id}")
        table_a = [k for k, v in display_map.items() if v == table_a_disp][0]

        table_b_disp = st.selectbox("Select Table B", [display_map[k] for k in short_names if k != table_a], key=f"table_b_{round_id}")
        table_b = [k for k, v in display_map.items() if v == table_b_disp][0]

        key_a = st.selectbox(f"Join key from {table_a}", tables[table_a].columns, key=f"key_a_{round_id}")
        key_b = st.selectbox(f"Join key from {table_b}", tables[table_b].columns, key=f"key_b_{round_id}")

        join_type = st.selectbox("Join type", ["inner", "left", "right", "outer", "cross"], key=f"join_type_{round_id}")

        na_option = st.selectbox(
            "How to handle missing values after join?",
            ["Do nothing", "Drop rows with NA", "Fill NA with 0", "Fill NA with mean"],
            key=f"na_option_{round_id}"
        )
        new_table_name = st.text_input("New table name", f"{table_a}_{table_b}", key=f"join_name_{round_id}")

        if st.button("Join Tables", key=f"join_btn_{round_id}"):
            # 🛠 Copy and convert join keys to str
            df1 = st.session_state.tables[table_a].copy()
            df2 = st.session_state.tables[table_b].copy()
            df1[key_a] = df1[key_a].astype(str)
            df2[key_b] = df2[key_b].astype(str)


            # Perform join
            if join_type == "cross":
                df_joined = df1.merge(df2, how="cross")
            else:
                df_joined = df1.merge(df2, left_on=key_a, right_on=key_b, how=join_type)

            # Handle missing values based on user selection
            if na_option == "Drop rows with NA":
                df_joined.dropna(inplace=True)
            elif na_option == "Fill NA with 0":
                df_joined.fillna(0, inplace=True)
            elif na_option == "Fill NA with mean":
                df_joined.fillna(df_joined.mean(numeric_only=True), inplace=True)

            # Replace with new table
            del st.session_state.tables[table_a]
            del st.session_state.tables[table_b]
            del st.session_state.table_name_map[table_a]
            del st.session_state.table_name_map[table_b]
            st.session_state.tables[new_table_name] = df_joined
            st.session_state.table_name_map[new_table_name] = new_table_name

            st.success(f"{new_table_name} created!")
            st.dataframe(df_joined.head())

            # Advance to next round
            st.session_state.join_round += 1

            # Ask to continue
            if len(tables) >= 2:
                cont = st.radio("Want to further join the table?", ["No", "Yes"], index=0, key=f"continue_{round_id}")
                if cont == "Yes":
                    st.session_state.keep_joining = True
            else:
                st.session_state.keep_joining = False

        break  # only render one round per rerun

    # Final output
    if st.session_state.tables:
        st.markdown("---")
        st.subheader("Results")
        keys = list(st.session_state.tables.keys())

        if len(keys) == 1:
            final = keys[0]
        else:
            final = st.selectbox("Choose final table", keys, key="final_table_select")

        df = pd.DataFrame(st.session_state.tables[final])
        st.info(f"✅ Table: {final}")
        st.dataframe(st.session_state.tables[final].head(10))
        
        csv = df.to_csv(index=False).encode('utf-8')
        col1, col2, _ = st.columns([1, 1, 1])

        with col1:
            if st.button("🔄 Reset All Tables"):
                for key in ["original_tables", "original_table_name_map", "tables", "table_name_map", "keep_joining"]:
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
        'SubBuilding': '',
        'BuildingNumber': '',
        'Street': '',
        'StreetType': '',
        'CityName': '',
        'ProvinceCode': 'BC',  # Always BC
        'PostalCode': '',
        'CountryName': ''
        }

        try:
            parts = [p.strip() for p in full_address.split(',') if p.strip()]
            if len(parts) >= 4:
                # Example: ["1666 W 75th Ave", "V6P 6G2", "Vancouver", "Canada"]
                street_part = parts[0]
                result['PostalCode'] = parts[1]
                result['CityName'] = parts[2]
                result['CountryName'] = parts[3]
            elif len(parts) == 3:
                street_part, result['PostalCode'], result['CityName'] = parts
                result['CountryName'] = "Canada"
            elif len(parts) == 2:
                street_part, result['PostalCode'] = parts
                result['CountryName'] = "Canada"
            else:
                street_part = parts[0]

            # Parse street line: e.g., "409 Granville St #256"
            match = re.match(
                r"(\d+)\s+([\w\s]+?)\s+(St|Ave|Street|Avenue|Blvd|Rd|Road|Dr|Drive|Way|Lane|Ln|Cres|Pl|Place|Court|Ct)\s*(#\d+)?",
                street_part.strip()
            )
            if match:
                result['BuildingNumber'] = match.group(1)
                result['Street'] = match.group(2).strip()
                result['StreetType'] = match.group(3)
                result['SubBuilding'] = match.group(4) if match.group(4) else ''

        except Exception as e:
            print(f"Fallback parsing failed: {e}")

        return result

    # --- Google API Key ---
    API_KEY = 'AIzaSyBW9Q1Vr3MWHx_xnHsInkd9xmOnGj7LhaE'  # Replace with your actual API key

    # --- Function to get coordinates ---
    def get_lat_lon(postal_code, country='Canada'):
        url = 'https://maps.googleapis.com/maps/api/geocode/json'
        params = {
            'address': f'{postal_code}, {country}',
            'key': API_KEY
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if data['status'] == 'OK':
                location = data['results'][0]['geometry']['location']
                return location['lat'], location['lng']
            else:
                print(f"[Google API] Error for {postal_code}: {data['status']}")
                return None, None
        except Exception as e:
            print(f"[Error] {postal_code}: {e}")
            return None, None
        
    st.subheader("🔎 Data Preprocessing")
    if st.checkbox("Choose the Address columns to elementize it."):
        selected_col = st.multiselect("filter Columns : ",df.columns, key="unique_select_vars")
        st.text(" ")

    if st.checkbox("Elementize the Addresses?"):
        df_tmp = df[selected_col]
        # Combine address lines into one
        #df["fullAddress"] = df_tmp.fillna("").agg(" ".join, axis=1).str.strip()
        df["fullAddress"] = df_tmp.fillna("").agg(lambda x: ", ".join([v for v in x if v]), axis=1).str.strip()
        api_key = "YA58-RG25-CX73-XD51"
        elementized_data = []
        parsed = {}
        for _, row in df.iterrows():
            full_address = row["fullAddress"]
            precision = row.get("PrecisionPoints", 0)
            
            if precision >= 99:
                try:
                    addr_id = find_address(api_key, full_address)
                    if addr_id:
                        parsed_raw = retrieve_full_address(api_key, addr_id)
                        parsed = {
                            "SubBuilding": parsed_raw.get("SubBuilding", ""),
                            "BuildingNumber": parsed_raw.get("BuildingNumber", ""),
                            "Street": parsed_raw.get("Street", ""),
                            "StreetType": parsed_raw.get("StreetType", ""),
                            "CityName": parsed_raw.get("City", ""),
                            "ProvinceCode": parsed_raw.get("ProvinceCode", ""),
                            "PostalCode": parsed_raw.get("PostalCode", ""),
                            "CountryName": parsed_raw.get("CountryName", "")
                        }
                    else:
                        parsed = {}
                        parsed = parse_address_fallback(full_address)
                except Exception:
                    parsed = {}
                    parsed = parse_address_fallback(full_address)
            else:
                # Fallback to basic parsing for low-precision data
                parsed = {}
                parsed = parse_address_fallback(full_address)
            elementized_data.append(parsed)
            #time.sleep(0.2)  # Avoid rate limits

        # --- Step 3: Merge parsed data and drop address lines ---
        address_df = pd.DataFrame(elementized_data)
        df.reset_index(drop=True, inplace=True)
        final_df = pd.concat([df, address_df], axis=1)
        final_df.drop(columns=["fullAddress"], errors="ignore", inplace=True)

        st.dataframe(final_df.head())
        df1 = final_df.copy()
        flag = 1
    else:
        df1 = df.copy()
    
    if st.checkbox("Get coordinate from an address postalcode?"):

        if(flag==1):
            temp_df = final_df.copy()
        else:
            temp_df = df.copy()

        
        # --- Ensure postalCode exists ---
        if 'PostalCode' not in temp_df.columns:
            raise ValueError("postalCode column not found in dataset.")

        # --- Get unique postal codes ---
        unique_postals = temp_df['PostalCode'].dropna().astype(str).unique()
        # --- Map postal codes to coordinates ---
        postal_to_coords = {}
        for pc in unique_postals:
            if pc.strip():
                lat, lon = get_lat_lon(pc)
                postal_to_coords[pc] = (lat, lon)
                time.sleep(0.2)  # Rate limiting

        # --- Append lat/lon to final_df ---
        final_df['Latitude'] = temp_df['PostalCode'].astype(str).map(lambda x: postal_to_coords.get(x, (None, None))[0])
        final_df['Longitude'] = temp_df['PostalCode'].astype(str).map(lambda x: postal_to_coords.get(x, (None, None))[1])
        st.dataframe(final_df.head())

       
        st.session_state.final_df = final_df.copy()

    else:
        df1 = df.copy()
        st.session_state.final_df = df1.copy()

# Descriptive Tab
with tabs[2]:
    st.subheader("📊 Descriptive Analysis")
        
    temp_df = st.session_state.final_df.copy()
    st.dataframe(temp_df.head())
    
    if st.checkbox("⏱️ Move Frequency Relative to Patient"): 
        move_freq = temp_df.groupby("ClientID").size().rename("MoveFrequency")
        temp_df = temp_df.merge(move_freq, on="ClientID", how="left")
        st.dataframe(temp_df)

    if st.checkbox("⚠️ Frequency of Missing or Incomplete Address Fields"):
        incomplete_fields = ['Street', 'StreetType', 'PostalCode', 'CityName']
        
        temp_df["row_has_missing"] = temp_df[incomplete_fields].isnull().any(axis=1)
        temp_df["MissingFrequency"] = temp_df.groupby("ClientID")["row_has_missing"].transform("sum")
        temp_df.drop(columns="row_has_missing", inplace=True)
        st.dataframe(temp_df)

    if st.checkbox("🏢 Classification: % Residential vs Addresses"):
        def classify_address(street):
            if pd.isna(street):
                return "Unknown"
            commercial_keywords = ["Office", "Centre", "Plaza", "Mall", "Business", "Tower"]
            if any(word.lower() in street.lower() for word in commercial_keywords):
                return "Commercial"
            return "Residential"

        temp_df["address_class"] = temp_df["Street"].apply(classify_address)
        
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

    if st.checkbox("📍 Distance Moved Between Address Changes"):
        def compute_move_distances(group):
            group = group.sort_values("StartDateID")
            coords = list(zip(group["Latitude"], group["Longitude"]))
            distances = [0]
            for i in range(1, len(coords)):
                if None not in coords[i] and None not in coords[i - 1]:
                    distances.append(geodesic(coords[i - 1], coords[i]).km)
                else:
                    distances.append(None)
            return pd.Series(distances, index=group.index)

        temp_df["MoveDistance(KM)"] = temp_df.groupby("ClientID").apply(compute_move_distances).reset_index(level=0, drop=True).round(1)
        st.dataframe(temp_df)

    final_df = temp_df.copy()

# Visualization Tab
with tabs[3]:
    st.subheader("📈 Data Visualization")
    

# Modeling Tab
with tabs[4]:
    st.subheader("💻 Machine Learning Models")
    

# Final fallback warning
if __name__ == '__main__':
    if not st.session_state.tables:
        st.warning("⚠️ Please upload a dataset to begin.")