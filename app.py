from flask import Flask, render_template, request, redirect, url_for, send_file
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, render_template_string, session, \
    jsonify
import datetime
from datetime import datetime
from datetime import datetime, timedelta
from datetime import datetime
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
import pyqrcode
import random
import os
import io
from datetime import datetime
import os
import smtplib
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import zipfile
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px

from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the path to the file for storing entries
entries_file = 'entries.csv'

app.secret_key = "secret_key"

# Folder to store uploaded files temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the folder for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'  # Change this to your desired folder path
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}  # Limit file types to CSV and Excel

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Load entries from the CSV file
def load_entries():
    if os.path.exists(entries_file):
        return pd.read_csv(entries_file).to_dict(orient='records')
    return []


# Save entries to the CSV file
def save_entries(entries):
    df = pd.DataFrame(entries)
    df.to_csv(entries_file, index=False)


# Home Route
@app.route('/')
def home():
    return render_template('home.html')


# # Dashboard Route
# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')


# API for Graph Data
# API for Graph Data

# DWM Report Routes
@app.route('/dwm_report')
def dwm_report():
    return render_template('dwm_report.html')


from datetime import datetime
import datetime

OPENING_CSV = "opening.csv"  # Opening data ke liye CSV file
CLOSING_CSV = "closing.csv"  # Closing data ke liye CSV file

def save_entry_to_csv(entry, filename):
    """Naye entry ko specified CSV file me append kare bina purane records ko overwrite kiye."""
    df = pd.DataFrame([entry])  # Entry ko DataFrame me convert karein
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))  # Append mode


@app.route('/dwm_report/opening', methods=['GET', 'POST'])
def opening():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['source'] = 'Opening'
        form_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Opening data ko opening.csv me store karein
        save_entry_to_csv(form_data, OPENING_CSV)

        return render_template('opening.html', message="Opening data submitted successfully!")

    return render_template('opening.html')





@app.route('/dwm_report/closing', methods=['GET', 'POST'])
def closing():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['source'] = 'Closing'
        form_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Closing data ko closing.csv me store karein
        save_entry_to_csv(form_data, CLOSING_CSV)

        return render_template('closing.html', message="Closing data submitted successfully!")

    return render_template('closing.html')


# DWM Row data dashboard
def load_and_merge_entries():
    """Opening aur Closing CSV files ko merge kare Date aur Shift columns ke basis par"""
    try:
        df_opening = pd.read_csv(OPENING_CSV)
        df_closing = pd.read_csv(CLOSING_CSV)
    except FileNotFoundError:
        print("Error: Opening ya Closing CSV file nahi mili!")
        return pd.DataFrame()  # Agar koi file nahi mili to empty DataFrame return karein

    # Ensure Date column is in datetime format
    df_opening["Date"] = pd.to_datetime(df_opening["Date"], errors='coerce')
    df_closing["Date"] = pd.to_datetime(df_closing["Date"], errors='coerce')

    # Check for NaT values in Date
    if df_opening["Date"].isna().sum() > 0 or df_closing["Date"].isna().sum() > 0:
        print("Warning: Kuch dates invalid hain, unhe NaT bana diya gaya hai!")

    # Columns jo suffix se exclude honge
    excluded_columns = {"Date", "Shift", "Location", "Customer"}

    # Opening aur Closing columns ke suffixes lagana
    df_opening = df_opening.rename(
        columns={col: f"{col}_opening" if col not in excluded_columns else col for col in df_opening.columns})
    df_closing = df_closing.rename(
        columns={col: f"{col}_closing" if col not in excluded_columns else col for col in df_closing.columns})

    # Merge on Date, Shift, Location, and Customer
    merged_df = pd.merge(df_opening, df_closing, on=["Date", "Shift", "Location", "Customer"], how="outer")

    # Final Output
    print("✅ Merging complete! Rows:", len(merged_df))

    return merged_df



@app.route('/dwm_report/dwm_data_dashboard', methods=['GET', 'POST'])
def dwm_data_dashboard():
    df = load_and_merge_entries()  # Load merged data

    if df.empty:
        return render_template('dwm_dashboard.html', data="<h3>No Data Available</h3>")

    # Default filter values
    current_filter = 'All'
    current_date = 'All'
    current_location = 'All'
    current_customer = 'All'

    # Filtering logic
    if request.method == 'POST':
        if 'clear_filters' in request.form:
            return render_template('dwm_dashboard.html',
                                   data=df.to_html(classes='table table-striped table-bordered', index=False),
                                   current_filter='All',
                                   current_date='All',
                                   current_location='All',
                                   current_customer='All')

        # Get form values
        current_filter = request.form.get('filter', 'All')
        current_date = request.form.get('Date', 'All')
        current_location = request.form.get('Location', 'All')
        current_customer = request.form.get('Customer', 'All')

        # Apply filters
        filtered_data = df.copy()

        # Date Filter
        if current_date != 'All' and current_date.strip():
            filtered_data = filtered_data[filtered_data["Date"] == current_date]

        # Location Filter
        if current_location != 'All':
            filtered_data = filtered_data[filtered_data['Location'] == current_location]

        # Customer Filter
        if current_customer != 'All':
            filtered_data = filtered_data[filtered_data['Customer'] == current_customer]

        return render_template('dwm_dashboard.html',
                               data=filtered_data.to_html(classes='table table-striped table-bordered', index=False),
                               current_filter=current_filter,
                               current_date=current_date,
                               current_location=current_location,
                               current_customer=current_customer)

    return render_template('dwm_dashboard.html',
                           data=df.to_html(classes='table table-striped table-bordered', index=False),
                           current_filter=current_filter,
                           current_date=current_date,
                           current_location=current_location,
                           current_customer=current_customer)


@app.route('/dwm_report/dwm_dashboard_ai', methods=['GET', 'POST'])
def dwm_dashboard_ai():
    df = load_and_merge_entries()

    # master_file = pd.read_csv(r"C:\Users\ashis\PycharmProjects\tral\master_data.csv")  # Master file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "master_data.csv")  # Correct Path

    try:
        master_file = pd.read_csv(file_path)  # ✅ Read CSV as DataFrame
    except FileNotFoundError:
        print(f"❌ Error: master_data.csv file not found at {file_path}")
        return "Error: master_data.csv file not found!", 500
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return f"Unexpected Error: {e}", 500

    if df.empty:
        return render_template('dwm_dashboard_ai.html', table_data=[], unique_dates=[], unique_shifts=[],
                               unique_locations=[], unique_customers=[])

    # Default filters
    current_date = request.form.get("Date", "")
    current_shift = request.form.get("Shift", "")
    current_location = request.form.get("Location", "")
    current_customer = request.form.get("Customer", "")

    # Apply filters if the form is submitted
    if request.method == 'POST':
        if 'clear_filters' in request.form:
            current_date, current_shift, current_location, current_customer = "", "", "", ""  # Reset filters
        else:
            if current_date:
                df = df[df["Date"] == current_date]
            if current_shift:
                df = df[df["Shift"] == current_shift]
            if current_location:
                df = df[df["Location"] == current_location]
            if current_customer:
                df = df[df["Customer"] == current_customer]

    # Benchmark Data
    benchmark_data = [
        {"Department": "Inward", "Activity": "Unloading / Loading Boxes", "Benchmark": 150},
        {"Department": "Inward", "Activity": "Receipt Process in WMS (Boxes)", "Benchmark": 150},
        {"Department": "Inward", "Activity": "Qty. GRN & QC", "Benchmark": 1100},
        {"Department": "Inventory", "Activity": "Qty - Good Putaway", "Benchmark": 1500},
        {"Department": "Inventory", "Activity": "Qty - Cycle Count/Consolidation", "Benchmark": 2000},
        {"Department": "Inventory", "Activity": "STN - Direct Putaway", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Qty. picked-B2C", "Benchmark": 400},
        {"Department": "Outward", "Activity": "QTY(Invoiced+Packed)B2C", "Benchmark": 450},
        {"Department": "Outward", "Activity": "QTY.Manifest+Handover-B2C", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Picked QTY. (B2B)", "Benchmark": 600},
        {"Department": "Outward", "Activity": "QTY(Invoiced+Packed)B2B", "Benchmark": 800},
        {"Department": "Return", "Activity": "RTO Received QTY. (B2B/B2C)", "Benchmark": 1000},
        {"Department": "Return", "Activity": "RTO Good processing return", "Benchmark": 200},
        {"Department": "Return", "Activity": "Bad processing with claim", "Benchmark": 60},
        {"Department": "Return", "Activity": "RTO Putway QTY.", "Benchmark": 1200},
        {"Department": "Return", "Activity": "Qty. GP Creation QCR", "Benchmark": 1000},
        {"Department": "Other Activities", "Activity": "", "Benchmark": 0},
    ]

    benchmarks_df = pd.DataFrame(benchmark_data)
    # print(benchmarks_df)


    # Manpower and Execution Calculations
    benchmarks_df['Deployed Manpower'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity} Manpower_opening", pd.Series(dtype='float64')).sum()
    ).round(2)  # 2 decimal places tak round karna

    benchmarks_df['Target'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity}_opening", pd.Series()).sum()
    )

    pendency_fields = [
        "GRN Qty_pendency", "STN Qty_pendency", "Putaway Cancel Qty_pendency", "Putaway Return Qty_pendency",
        "GRN Sellable Qty_pendency", "Bin Movement_pendency", "Return_pendency", "RTV_pendency",
        "Channel Order Qty (B2C)_pendency", "RTS Order Qty (B2C)_pendency", "Breached Qty_pendency",
        "Side Lined_pendency", "Dispatch Not Marked", "Not Dispatched Orders"
    ]

    for field in pendency_fields:
        field_name = f"{field}_opening"
        benchmarks_df['Target'] += df.get(field_name, pd.Series()).sum()

    # Required Manpower Calculation
    benchmarks_df['Required Manpower'] = (benchmarks_df['Target'] / benchmarks_df['Benchmark']).round(2)
    benchmarks_df['Extra Manpower'] = benchmarks_df['Deployed Manpower'] - benchmarks_df['Required Manpower'].round(2)

    benchmarks_df['Execution'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity}_closing", pd.Series()).sum()
    )
    benchmarks_df['Pendency'] = benchmarks_df['Target'] - benchmarks_df['Execution']

    benchmarks_df['Capacity'] = benchmarks_df['Deployed Manpower'] * benchmarks_df['Benchmark']

    benchmarks_df['Capacity Vs Execution'] = benchmarks_df.apply(
        lambda row: f"{(row['Execution'] / row['Capacity'] * 100) if row['Capacity'] > 0 else 0:.2f}%", axis=1
    )

    benchmarks_df['Target Vs Execution'] = benchmarks_df.apply(
        lambda row: f"{(row['Execution'] / row['Target'] * 100) if row['Target'] > 0 else 0:.2f}%", axis=1
    )

    table_data_list = benchmarks_df.to_dict(orient='records')

    # SECOND MERGE: Merge table_data_list AGAIN with master file
    table_data_df = pd.DataFrame(table_data_list)  # Convert to DataFrame
    table_data_df = table_data_df.merge(master_file, on="Activity", how="left")  # Second merge

    # Rename Columns
    table_data_df.rename(columns={
        'Department_x': 'Department',
        'Benchmark_x': 'Benchmark',
        'Planned Load': 'Planned Load',
        'Head Count': 'Head Count'
    }, inplace=True)
    # Reorder Columns
    column_order = [
        'Department', 'Activity', 'Benchmark', 'Head Count', 'Planned Load', 'Pendency','Deployed Manpower',
        'Target', 'Required Manpower', 'Extra Manpower', 'Execution',
        'Capacity','Capacity Vs Execution', 'Target Vs Execution']
    table_data_df = table_data_df[column_order]  # Reordering

    # Convert back to list after second merge
    final_table_data = table_data_df.to_dict(orient='records')

    # Store merged data for download
    session['table_data_list'] = final_table_data

    return render_template("dwm_dashboard_ai.html", table_data=final_table_data,
                           unique_dates=df["Date"].unique().tolist(),
                           unique_shifts=df["Shift"].unique().tolist(),
                           unique_locations=df["Location"].unique().tolist(),
                           unique_customers=df["Customer"].unique().tolist(),
                           selected_date=current_date, selected_shift=current_shift,
                           selected_location=current_location, selected_customer=current_customer)


@app.route("/download")
def download_report():
    df = pd.DataFrame(session.get('table_data_list', []))

    if df.empty:
        return "No data to download"

    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="DWM_Report.csv")



# SLA Report Route
@app.route('/sla_report')
def sla_report():
    return render_template('sla_report.html')

from datetime import datetime

# Folder to save reports
REPORTS_FOLDER = r"C:\Users\ashis\OneDrive\Desktop\humen\12192024"
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)


@app.route('/sla_report/unicom', methods=['GET', 'POST'])
def unicom():
    report_files = []
    table_html = []

    if request.method == 'POST':
        uploaded_files = request.files.getlist("file")

        if not uploaded_files:
            return "No file uploaded", 400

        for file in uploaded_files:
            if file.filename == '':
                continue

            file_path = os.path.join(REPORTS_FOLDER, file.filename)
            file.save(file_path)

            try:
                df = pd.read_csv(file_path, low_memory=False)

                df['Fulfillment TAT'] = pd.to_datetime(df['Fulfillment TAT'], errors='coerce')
                df['Invoice Created'] = pd.to_datetime(df['Invoice Created'], errors='coerce')
                df['Fulfillment TAT_DATE'] = df['Fulfillment TAT'].dt.date
                df['Fulfillment TAT_DATE'] = pd.to_datetime(df['Fulfillment TAT_DATE'], errors='coerce')

                yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()

                df = df[df["Fulfillment TAT_DATE"] <= yesterday]
                df['SLA'] = df.apply(lambda x: "Cancelled" if x['Sale Order Item Status'] == "CANCELLED"
                                     else "Within SLA" if x['Invoice Created'] <= x['Fulfillment TAT']
                                     else "SLA Breached", axis=1)

                facilities = ["Kothari_HYD", "Kothari_GGN"]
                report_types = ["Yesterday", "Overall"]

                for facility in facilities:
                    for report_type in report_types:
                        facility_df = df[df["Facility"] == facility]

                        pivot_table = facility_df.pivot_table(values='Sale Order Item Code',
                                                              index='Channel Name',
                                                              columns='SLA',
                                                              aggfunc='count',
                                                              fill_value=0)

                        pivot_table = pivot_table.rename(columns={
                            "Cancelled": "Cancelled",
                            "Within SLA": "With in SLA",
                            "SLA Breached": "SLA Breached"
                        })

                        for col in ["Cancelled", "With in SLA", "SLA Breached"]:
                            if col not in pivot_table.columns:
                                pivot_table[col] = 0

                        pivot_table["No. of orders"] = pivot_table.sum(axis=1)

                        total_row = pivot_table.sum(numeric_only=True)
                        total_row.name = "Total"
                        pivot_table = pd.concat([pivot_table, total_row.to_frame().T])

                        pivot_table["SLA Breached %"] = (pivot_table["SLA Breached"] / pivot_table["No. of orders"]) * 100
                        pivot_table["SLA Breached %"] = pivot_table["SLA Breached %"].fillna(0).map(lambda x: f"{x:.2f}%")

                        pivot_table.reset_index(inplace=True)
                        pivot_table.rename(columns={"Channel Name": "Sales Channels"}, inplace=True)

                        report_filename = f"{report_type}_{facility}_Export_Sale_Report_{datetime.now().strftime('%Y%m%d')}.csv"
                        report_path = os.path.join(REPORTS_FOLDER, report_filename)
                        pivot_table.to_csv(report_path, index=False)
                        report_files.append(report_filename)

                        table_html.append({
                            "facility": facility,
                            "report_type": report_type,
                            "table_html": pivot_table.to_html(classes='table table-striped table-hover', escape=False, index=False)
                        })

            except Exception as e:
                return f"Error processing file: {e}", 500

    return render_template('unicom.html', reports=report_files, now=datetime.now(), tables=table_html)


@app.route('/download/<filename>')
def download_report_uni(filename):
    file_path = os.path.join(REPORTS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


# Home Page & Upload Logic
# Allowed extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Page & Upload Logic
@app.route('/sla_report/eshopbox', methods=['GET', 'POST'])
def eshopbox():
    report_files = []

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists
            file.save(file_path)

            # Process the file
            report_files = process_eshopbox_report(file_path)

    return render_template('eshopbox.html', reports=report_files)

# Processing Eshopbox SLA Report
def process_eshopbox_report(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    # Convert necessary columns to datetime
    date_cols = ['Shipment created in flex', 'Expected RTS at', 'Packed at', 'Shipment dispatched at']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Converts invalid dates to NaT

    # Check if any column contains non-datetime values
    for col in date_cols:
        print(f"{col} - Non-Date Values:", df[col][~df[col].apply(pd.api.types.is_datetime64_any_dtype)])

    # Filter for shipments created before yesterday
    Yesterday_Date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()

    df = df.dropna(subset=['Shipment created in flex', 'Packed at'])  #

    df['Shipment created in Eshopbox'] = pd.to_datetime(df['Shipment created in Eshopbox'], errors='coerce')
    df['Packed at'] = pd.to_datetime(df['Packed at'], errors='coerce')
    df['Expected RTS at'] = pd.to_datetime(df['Expected RTS at'], errors='coerce')

    df['Shipment created DATE'] = df['Shipment created in Eshopbox'].dt.date
    # df = df[df["Shipment created DATE"] <= yesterday]
    df['Shipment created in flex_MONTH'] = df['Shipment created in flex'].dt.strftime('%b').str.upper()
    df['Shipment created in flex_DATE'] = df['Shipment created in flex'].dt.date
    df['Shipment created in flex_DATE'] = pd.to_datetime(df['Shipment created in flex_DATE'], errors='coerce')
    df['Shipment dispatched at_DATE'] = df['Shipment dispatched at'].dt.date
    df['Shipment dispatched at_DATE'] = pd.to_datetime(df['Shipment dispatched at_DATE'], errors='coerce')
    df['Expected RTS at_DATE'] = df['Expected RTS at'].dt.date
    df['Expected RTS at_DATE'] = pd.to_datetime(df['Expected RTS at_DATE'], errors='coerce')
    df['Packed at_DATE'] = df['Packed at'].dt.date
    df['Packed at_DATE'] = pd.to_datetime(df['Packed at_DATE'], errors='coerce')
    df = df[df["Expected RTS at_DATE"] <= Yesterday_Date]
    df = df[df["Shipment status"] != "HOLD"]

    cutoff_time = pd.to_datetime('16:00:00').time()
    Amazon = ["FBA", "COCOBLU_AGGN_BEING_HUMAN", "COCOBLU_SPYKAR_ZMUM", "MFN", "Amazon", "COCOBLU_AHYD_BEING_HUMAN"]
    df['Handover'] = df.apply(lambda x: "Handover in SLA" if (
            x['Shipment created in flex_DATE'] == x['Shipment dispatched at_DATE'] and
            (x['Shipment dispatched at'].time() <= cutoff_time if x['Sales channel'] in Amazon else True)
    ) else "Handover breached", axis=1)

    AJIOandB2C = ["Ajio", "Ajio Raymond Lifestyle", "Nykaa", "Nykaa com", "Nykaa Fashion (New)",
                  "Nykaa Fashion Being Hyman YGGN", "Nykaa Fashion Spykar", "Nykaa_com", "NYKAA_MAN_SPYKAR_ZMUM",
                  "NykaaDotRaymondLifestyleLimited", "NykaaFashionRaymondLifestyleLimited", "SNAPDEAL_DUKE_AGGN",
                  "SNAPDEAL_ZMUM_DUKE"]
    tata_cliq_channels = ["Tata Cliq", "Tata Cliq Raymond Lifestyle Limited", "Tata Cliq Grasim", "Tata Cliq Lux",
                          "Tata Cliq Being Human"]



    df1 = df[~df["Sales channel"].isin(AJIOandB2C + tata_cliq_channels)]

    # SLA Calculation

    df1['SLA'] = df1.apply(lambda x: "Cancelled" if x['Shipment status'] == "CANCELLED"
    else "Tech error" if pd.notnull(x['Label error message'])
    else "With in SLA" if x['Packed at'] <= x['Expected RTS at']
    else "SLA breached", axis=1)

    filtered_df1_by_Yesterday = df1[(df1['Expected RTS at_DATE'] == Yesterday_Date) &
                                    ((df1['Packed at_DATE'] >= Yesterday_Date) | (df1['Packed at_DATE']).isnull())]

    df2 = df[df["Sales channel"].isin(AJIOandB2C)]
    cutoff_time1 = pd.to_datetime('12:00:00').time()
    cutoff_time2 = pd.to_datetime('16:00:00').time()

    df2['SLA'] = "x"

    # Iterate through the DataFrame
    for index, row in df2.iterrows():
        if row['Shipment status'] == "CANCELLED":
            df2.at[index, 'SLA'] = "Cancelled"
        elif row['Shipment status'] in ["SIDELINED ON PACK", "SIDELINED ON HANDOVER"]:
            df2.at[index, 'SLA'] = "Tech error"
        elif (row['Shipment created in flex'].date() == row['Packed at'].date() and
              row['Shipment created in flex'].time() < cutoff_time1 and
              row['Packed at'].time() <= cutoff_time2):
            df2.at[index, 'SLA'] = "With in SLA"
        elif (row['Shipment created in flex'].date() == row['Packed at'].date() - pd.Timedelta(days=1) and
              row['Shipment created in flex'].time() >= cutoff_time1 and
              row['Packed at'].time() < cutoff_time2):
            df2.at[index, 'SLA'] = "With in SLA"
        elif (row['Shipment created in flex'].date() == row['Packed at'].date() and
              row['Shipment created in flex'].time() >= cutoff_time1):
            df2.at[index, 'SLA'] = "With in SLA"
        else:
            df2.at[index, 'SLA'] = "SLA breached"

    filtered_df2_by_Yesterday = df2[((df2["Shipment created in flex_DATE"] == Yesterday_Date) & (
                df2['Shipment created in flex'].dt.time < cutoff_time1)) |
                                    ((df2["Shipment created in flex_DATE"] == Yesterday_Date) & (
                                                df2['Shipment created in flex'].dt.time >= cutoff_time1)
                                     & (df2['Packed at_DATE'] == Yesterday_Date)) |
                                    ((df2["Shipment created in flex_DATE"] == (Yesterday_Date - pd.Timedelta(days=1))) &
                                     (df2['Shipment created in flex'].dt.time >= cutoff_time1) &
                                     ((df2['Packed at_DATE'] >= Yesterday_Date) | (df2['Packed at_DATE'].isnull())))]

    df3 = df[df["Sales channel"].isin(tata_cliq_channels)]
    cutoff_time3 = pd.to_datetime('14:15:00').time()

    df3['SLA'] = "x"

    # Iterate through the DataFrame
    for index, row in df3.iterrows():
        if row['Shipment status'] == "CANCELLED":
            df3.at[index, 'SLA'] = "Cancelled"
        elif row['Shipment status'] in ["SIDELINED ON PACK", "SIDELINED ON HANDOVER"]:
            df3.at[index, 'SLA'] = "Tech error"
        elif (row['Shipment created in flex'].date() == row['Packed at'].date() and
              row['Shipment created in flex'].time() < cutoff_time3 and
              row['Packed at'].time() <= cutoff_time3):
            df3.at[index, 'SLA'] = "With in SLA"
        elif (row['Shipment created in flex'].date() == row['Packed at'].date() - pd.Timedelta(days=1) and
              row['Shipment created in flex'].time() >= cutoff_time3 and
              row['Packed at'].time() <= cutoff_time3):
            df3.at[index, 'SLA'] = "With in SLA"
        else:
            df3.at[index, 'SLA'] = "SLA breached"

    filtered_df3_by_Yesterday = df3[((df3["Shipment created in flex_DATE"] == Yesterday_Date) & (
                df3['Shipment created in flex'].dt.time < cutoff_time3)) |
                                    ((df3["Shipment created in flex_DATE"] == Yesterday_Date) & (
                                                df3['Shipment created in flex'].dt.time >= cutoff_time3)
                                     & (df3['Packed at_DATE'] == Yesterday_Date)) |
                                    ((df3["Shipment created in flex_DATE"] == (Yesterday_Date - pd.Timedelta(days=1))) &
                                     (df3['Shipment created in flex'].dt.time >= cutoff_time3) &
                                     ((df3['Packed at_DATE'] >= Yesterday_Date) | (df3['Packed at_DATE'].isnull())))]

    # Yesterday SLA Report
    df_Yesterday = pd.concat([filtered_df1_by_Yesterday, filtered_df2_by_Yesterday, filtered_df3_by_Yesterday])
    pivot_table_yest = df_Yesterday.pivot_table(values='Order item IDs', index='Sales channel', columns='SLA',
                                                aggfunc='sum', fill_value=0)
    pivot_table_yest['Order Quantity'] = pivot_table_yest.sum(axis=1)
    pivot_table_yest = pivot_table_yest.astype(int)
    pivot_table_yest_2 = df_Yesterday.pivot_table(values='Order item IDs', index='Sales channel', aggfunc='count',
                                                  fill_value=0)
    pivot_table_yest_2 = pivot_table_yest_2.rename(columns={"Order item IDs": "No. of orders"})

    pivot_table_yest_3 = df_Yesterday.pivot_table(values='Order item IDs', index='Sales channel', columns='Handover',
                                                  aggfunc='sum', fill_value=0)

    pivot_table_yest_3 = pivot_table_yest_3.rename(columns={"Order item IDs": "No. of orders"})

    pivot_table_yest = pd.concat([pivot_table_yest, pivot_table_yest_3], axis=1)
    total_row = pivot_table_yest.sum(numeric_only=True)
    total_row.name = 'Total'  # Set the name for the total row

    # pivot_table_yest = pivot_table_yest.fillna(0).astype(int)


    # Use pd.concat to add total row
    pivot_table_yest = pd.concat([pivot_table_yest, total_row.to_frame().T])

    # Calculate total percentage row
    total_percentage = (pivot_table_yest.loc['Total'] / pivot_table_yest.loc['Total']['Order Quantity']) * 100
    total_percentage.name = 'Total Percentage'

    # # # Append the total percentage row
    pivot_table_yest = pd.concat([pivot_table_yest, total_percentage.to_frame().T])
    pivot_table_yest = pivot_table_yest.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    Yest_SLA_Report = pd.concat([pivot_table_yest, pivot_table_yest_2], axis=1)
    Yest_SLA_Report = Yest_SLA_Report.fillna(0)
    Yest_SLA_Report = Yest_SLA_Report.astype(int)

    sla_breached = Yest_SLA_Report.get('SLA breached', 0)
    Yest_SLA_Report['SLA Breached %'] = (sla_breached / Yest_SLA_Report['Order Quantity']) * 100
    Yest_SLA_Report['SLA Breached %'] = Yest_SLA_Report['SLA Breached %'].fillna(0)  # Rep

    Yest_SLA_Report['SLA Breached %'] = Yest_SLA_Report['SLA Breached %'].map(lambda x: f"{x:.2f}")
    Yest_SLA_Report.index.name = "Sales Channels"

    # Total orders
    Yest_SLA_Report.columns = Yest_SLA_Report.columns.str.strip()


    Yest_SLA_Report.loc["Total", "No. of orders"] = Yest_SLA_Report["No. of orders"].sum()

    # Yest_SLA_Report.loc["Total", "No. of orders"] = Yest_SLA_Report["No. of orders"].sum()
    # print(Yest_SLA_Report.columns)

    # Yest_SLA_Report.loc["Total", "No. of orders"] = Yest_SLA_Report["No. of orders"].sum()

    Yest_SLA_Report = Yest_SLA_Report[
        ['Cancelled', 'Tech error', 'With in SLA', 'SLA breached', 'Order Quantity', 'No. of orders', 'Handover in SLA',
         'Handover breached', 'SLA Breached %']]

    df = pd.concat([df1, df2, df3])
    pivot_table1 = df.pivot_table(values='Order item IDs', index='Sales channel', columns='SLA', aggfunc='sum',
                                  fill_value=0)
    pivot_table1['Order Quantity'] = pivot_table1.sum(axis=1)
    pivot_table1 = pivot_table1.astype(int)

    pivot_table2 = df.pivot_table(values='Order item IDs', index='Sales channel', aggfunc='count', fill_value=0)
    pivot_table2 = pivot_table2.rename(columns={"Order item IDs": "No. of orders"})
    pivot_table3 = df.pivot_table(values='Order item IDs', index='Sales channel', columns='Handover', aggfunc='sum',
                                  fill_value=0)

    pivot_table = pd.concat([pivot_table1, pivot_table3], axis=1)
    total_row = pivot_table.sum(numeric_only=True)
    total_row.name = 'Total'  # Set the name for the total row

    # Use pd.concat to add total row
    pivot_table = pd.concat([pivot_table, total_row.to_frame().T])

    # Calculate total percentage row
    total_percentage = (pivot_table.loc['Total'] / pivot_table.loc['Total']['Order Quantity']) * 100
    total_percentage.name = 'Total Percentage'

    # # # Append the total percentage row
    pivot_table = pd.concat([pivot_table, total_percentage.to_frame().T])
    pivot_table = pivot_table.astype(int)

    SLA_Report = pd.concat([pivot_table, pivot_table2], axis=1)
    SLA_Report = SLA_Report.fillna(0)
    SLA_Report = SLA_Report.astype(int)

    sla_breached = SLA_Report.get('SLA breached', 0)
    SLA_Report['SLA Breached %'] = (sla_breached / SLA_Report['Order Quantity']) * 100
    SLA_Report['SLA Breached %'] = SLA_Report['SLA Breached %'].fillna(0)  # Rep

    SLA_Report['SLA Breached %'] = SLA_Report['SLA Breached %'].map(lambda x: f"{x:.2f}")
    SLA_Report.index.name = "Sales Channels"
    SLA_Report.loc["Total", "No. of orders"] = SLA_Report["No. of orders"].sum()

    SLA_Report = SLA_Report[
        ['Cancelled', 'Tech error', 'With in SLA', 'SLA breached', 'Order Quantity', 'No. of orders', 'Handover in SLA',
         'Handover breached', 'SLA Breached %']]

    return [Yest_SLA_Report, SLA_Report]


# Download Report
@app.route('/download_eshopbox/<filename>')
def download_eshopbox_report(filename):
    file_path = os.path.join("temp", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/sla_report/pendency_sla', methods=['GET', 'POST'])
def pendency_sla():

    return render_template('pendency_sla.html')




# KPI Report Route
@app.route('/kpi_report')
def kpi_report():
    return render_template('kpi_report.html')


# HRMS Report Route
@app.route('/hrms_report')
def hrms_report():
    return render_template('hrms_report.html')


# Unicom Report Route
@app.route('/unicom_report')
def unicom_report():
    return render_template('unicom_report.html')


# Download Dashboard Route


if __name__ == '__main__':
    app.run(debug=True)
