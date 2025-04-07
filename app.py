<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, render_template_string, session, \
    jsonify
import datetime
from datetime import datetime, timedelta
from datetime import datetime
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
import pyqrcode
import random
import os
import io
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


# Dashboard Route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# API for Graph Data
# API for Graph Data
@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    df = load_entries()

    if df.empty:
        return jsonify({"error": "No data available"})

    # Data Aggregation for Charts
    location_counts = df["Location"].value_counts().to_dict()
    customer_counts = df["Customer"].value_counts().to_dict()

    shift_counts = df["Shift"].value_counts().to_dict()
    inward_counts = df["Inward Name_opening"].value_counts().to_dict()

    # Response Data
    return jsonify({
        "locations": location_counts,
        "customers": customer_counts,
        "shifts": shift_counts,
        "inwards": inward_counts
    })


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
        form_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Opening data ko opening.csv me store karein
        save_entry_to_csv(form_data, OPENING_CSV)

        return render_template('opening.html', message="Opening data submitted successfully!")

    return render_template('opening.html')

@app.route('/dwm_report/closing', methods=['GET', 'POST'])
def closing():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['source'] = 'Closing'
        form_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        return pd.DataFrame()  # Agar koi file nahi mili to empty DataFrame return karein
        # Ensure Date column is in datetime format
    df_opening["Date"] = pd.to_datetime(df_opening["Date"], errors='coerce')
    df_closing["Date"] = pd.to_datetime(df_closing["Date"], errors='coerce')
    # Suffix lagane ke liye excluded columns
    excluded_columns = {"Date", "Shift", "Location", "Customer"}

    # Opening aur Closing columns ke suffixes
    df_opening = df_opening.rename(
        columns={col: f"{col}_opening" if col not in excluded_columns else col for col in df_opening.columns}
    )
    df_closing = df_closing.rename(
        columns={col: f"{col}_closing" if col not in excluded_columns else col for col in df_closing.columns}
    )

    # Merge on Date and Shift
    merged_df = pd.merge(df_opening, df_closing, on=["Date", "Shift", "Location", "Customer"], how="outer")

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
        {"Department": "Inventory", "Activity": "Qty- Good Putaway", "Benchmark": 1500},
        {"Department": "Inventory", "Activity": "Qty- Cycle Count/Consolidation", "Benchmark": 2000},
        {"Department": "Inventory", "Activity": "STN-Direct Putaway", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Qty. picked-B2C", "Benchmark": 400},
        {"Department": "Outward", "Activity": "QTY(Invoiced+Packed)B2C", "Benchmark": 450},
        {"Department": "Outward", "Activity": "QTY.Manifest+Handover-B2C", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Picked QTY. ( B2B )", "Benchmark": 600},
        {"Department": "Outward", "Activity": "QTY(Invoiced+Packed)B2B", "Benchmark": 800},
        {"Department": "Return", "Activity": "RTO Received QTY. (B2B/B2C)", "Benchmark": 1000},
        {"Department": "Return", "Activity": "RTO Good processing return", "Benchmark": 200},
        {"Department": "Return", "Activity": "Bad processing with claim", "Benchmark": 60},
        {"Department": "Return", "Activity": "RTO Putaway", "Benchmark": 1200},
        {"Department": "Return", "Activity": "Qty. GP Creation QCR", "Benchmark": 1000},
        {"Department": "Other Activities", "Activity": "", "Benchmark": 0},
    ]

    benchmarks_df = pd.DataFrame(benchmark_data)

    # Manpower and Execution Calculations
    benchmarks_df['Deployed Manpower'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity} Manpower_opening", pd.Series()).sum()
    )

    benchmarks_df['Target'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity}_opening", pd.Series()).sum()
    )

    # Required Manpower Calculation
    benchmarks_df['Required Manpower'] = (benchmarks_df['Target'] / benchmarks_df['Benchmark']).round(2)
    benchmarks_df['Extra Manpower'] = benchmarks_df['Deployed Manpower'] - benchmarks_df['Required Manpower'].round(2)

    benchmarks_df['Execution'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity}_closing", pd.Series()).sum()
    )

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
        'Department', 'Activity', 'Benchmark', 'Head Count', 'Deployed Manpower',
        'Target', 'Required Manpower', 'Extra Manpower', 'Execution',
        'Capacity','Planned Load', 'Capacity Vs Execution', 'Target Vs Execution']
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


@app.route('/dwm_report/download_dashboard_ai', methods=['GET'])
def download_dashboard_ai():
    # Fetch the filtered data (same as passed to the template)
    table_data = session.get('table_data', None)

    if not table_data:
        return "No data available for download", 400

    # Convert the list of records (table data) back to a DataFrame
    df_for_download = pd.DataFrame(table_data)

    # Apply the same filters that were applied on the dashboard page
    selected_date = request.args.get('Date', default=None)
    selected_shift = request.args.get('Shift', default=None)
    selected_location = request.args.get('Location', default=None)
    selected_customer = request.args.get('Customer', default=None)

    # Apply filters to the DataFrame
    if selected_date:
        df_for_download = df_for_download[df_for_download["Date"] == selected_date]
    if selected_shift:
        df_for_download = df_for_download[df_for_download["Shift"] == selected_shift]
    if selected_location:
        df_for_download = df_for_download[df_for_download["Location"] == selected_location]
    if selected_customer:
        df_for_download = df_for_download[df_for_download["Customer"] == selected_customer]

    # Define the correct column order to match the dashboard display
    column_order = [
        'Department', 'Activity', 'Benchmark', 'Deployed Manpower',
        'Required Manpower', 'Extra Manpower', 'Target', 'Capacity',
        'Execution', 'Capacity Vs Execution', 'Target Vs Execution'
    ]

    # Reorder the columns in the DataFrame
    df_for_download = df_for_download[column_order]

    # Convert the filtered DataFrame to CSV format
    csv_data = df_for_download.to_csv(index=False)

    # Create an in-memory buffer for the CSV file
    buffer = io.BytesIO()
    buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)

    # Send the filtered CSV file as a response
    return send_file(buffer, as_attachment=True, download_name="filtered_dwm_dashboard_data.csv", mimetype="text/csv")


def calculate_required_manpower(value):
    """
    Example function to calculate required manpower.
    Adjust the logic based on actual requirements.
    """
    try:
        return int(value) * 2  # Example: Required manpower is twice the value
    except (ValueError, TypeError):
        return "N/A"


# Project Report Route
# Project Report Route


# Load initial warehouse data (Simulated function)
def load_entries():
    return session.get('table_data', [])


MASTER_FILE_PATH = os.path.join(UPLOAD_FOLDER, 'master_data.xlsx')


def load_master_data():
    if os.path.exists(MASTER_FILE_PATH):
        master_df = pd.read_excel(MASTER_FILE_PATH, engine='openpyxl')
        required_columns = {"Department", "Activity", "Benchmark", "Planned Load", "Head Count"}
        if required_columns.issubset(master_df.columns):
            return master_df
    return pd.DataFrame(columns=["Department", "Activity", "Benchmark", "Planned Load", "Head Count"])


@app.route('/upload_master', methods=['POST'])
def upload_master():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    file.save(MASTER_FILE_PATH)
    return redirect(url_for('projection_dashboard'))


@app.route('/dwm_report/projection_dashboard', methods=['GET'])
def projection_dashboard():
    table_data_list = session.get('table_data', [])
    df = pd.DataFrame(table_data_list)
    master_df = load_master_data()

    if not df.empty:
        df = df.merge(master_df, on="Activity", how="left", suffixes=("", "_master"))
        for col in ["Benchmark", "Planned Load", "Head Count"]:
            if col in df.columns and f"{col}_master" in df.columns:
                df[col] = df[f"{col}_master"].fillna(df[col])
                df.drop(columns=[f"{col}_master"], inplace=True)

    filters = {key: request.args.get(key) for key in ["Date", "Shift", "Location", "Customer"] if request.args.get(key)}
    for key, value in filters.items():
        df = df[df[key] == value]

    table_data_list = df.to_dict(orient='records')
    unique_filters = {col: df[col].dropna().unique().tolist() if col in df.columns else [] for col in
                      ["Date", "Shift", "Location", "Customer"]}

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Warehouse Projection Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 20px;
                    padding: 20px;
                    text-align: center;
                }
                h2 {
                    color: #333;
                }
                form {
                    margin-bottom: 20px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    background: #fff;
                }
                th, td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                th {
                    background-color: #007bff;
                    color: white;
                }
                select, button {
                    padding: 8px;
                    margin: 5px;
                    border-radius: 5px;
                }
                button {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <h2>Upload Master File</h2>
            <form action="/upload_master" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".xlsx">
                <button type="submit">Upload</button>
            </form>
            <h2>Filter Data</h2>
            <form method="GET">
                {% for key, values in unique_filters.items() %}
                <label>{{ key }}:</label>
                <select name="{{ key }}">
                    <option value="">--Select--</option>
                    {% for value in values %}
                    <option value="{{ value }}">{{ value }}</option>
                    {% endfor %}
                </select>
                {% endfor %}
                <button type="submit">Apply</button>
            </form>
            <h2>Dashboard Data</h2>
            {% if table_data_list %}
            <table>
                <tr>
                    {% for col in table_data_list[0].keys() %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
                {% for row in table_data_list %}
                <tr>
                    {% for value in row.values() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No data available.</p>
            {% endif %}
        </body>
        </html>
    """, table_data_list=table_data_list, unique_filters=unique_filters)


# @dash_app.server.route('/projection_dashboard')
# def projection_dashboard():
#     return render_template('projection_dashboard.html')


# SLA Report Route
@app.route('/sla_report')
def sla_report():
    return render_template('sla_report.html')


ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/sla_report/unicom', methods=['GET', 'POST'])
def unicom():
    try:
        if request.method == 'POST':
            # Check if the request contains a file
            if 'file' not in request.files:
                return "No file part in the request", 400

            file = request.files['file']

            # Check if a file is selected
            if file.filename == '':
                return "No file selected", 400

            # Validate and save the file
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Load the file into a DataFrame
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)
                else:
                    return "Invalid file format. Allowed formats are CSV and Excel.", 400

                # Ensure the dataframe is not empty
                if df.empty:
                    return "Uploaded file contains no data.", 400

                # Preprocess the dataframe
                df.columns = df.columns.str.upper().str.replace(" ", "_")
                df['FULFILLMENT_TAT'] = pd.to_datetime(df['FULFILLMENT_TAT'], errors='coerce')
                df['INVOICE_CREATED'] = pd.to_datetime(df['INVOICE_CREATED'], errors='coerce')

                # Check if required columns exist
                required_columns = ['FULFILLMENT_TAT', 'INVOICE_CREATED', 'SALE_ORDER_ITEM_STATUS', 'FACILITY',
                                    'CHANNEL_NAME']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    return f"Missing required columns: {', '.join(missing_columns)}", 400

                # Filter for yesterday's data
                yesterday_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize().date()
                df['FULFILLMENT_TAT_DATE'] = pd.to_datetime(df['FULFILLMENT_TAT'].dt.date, errors='coerce')
                df = df[df['FULFILLMENT_TAT_DATE'] <= pd.Timestamp(yesterday_date)]

                # Add SLA status
                df['SLA'] = df.apply(
                    lambda x: "Cancelled" if x['SALE_ORDER_ITEM_STATUS'] == "CANCELLED"
                    else "With in SLA" if x['INVOICE_CREATED'] <= x['FULFILLMENT_TAT']
                    else "SLA breached", axis=1)

                # Filter data for yesterday
                filtered_df = df[(
                                         df['FULFILLMENT_TAT_DATE'] == pd.Timestamp(yesterday_date)) &
                                 ((df['INVOICE_CREATED'] >= pd.Timestamp(yesterday_date)) | (
                                     df['INVOICE_CREATED'].isnull()))
                                 ]

                # Ensure there is data after filtering
                if filtered_df.empty:
                    return "No data available for the selected date range.", 400

                # Create pivot tables for specified facilities
                pivot_tables = {}
                facilities = ['Kothari_HYD', 'Kothari_GGN']

                required_sla_columns = ['Cancelled', 'SLA breached', 'With in SLA', 'No. of orders', 'SLA Breached %']

                for facility in facilities:
                    facility_data = filtered_df[filtered_df['FACILITY'] == facility]

                    # First Pivot Table (Yesterday's Data)
                    pivot_table_yesterday = facility_data.pivot_table(
                        values='SALE_ORDER_ITEM_STATUS',
                        index='CHANNEL_NAME',
                        columns='SLA',
                        aggfunc='count',
                        fill_value=0
                    )

                    # Ensure all required SLA columns are present
                    for col in required_sla_columns[:-2]:  # Exclude 'No. of orders' and 'SLA Breached %' from this loop
                        if col not in pivot_table_yesterday.columns:
                            pivot_table_yesterday[col] = 0

                    # Add the 'No. of orders' column
                    pivot_table_yesterday['No. of orders'] = pivot_table_yesterday.sum(axis=1)

                    # Add the 'SLA Breached %' column
                    if 'SLA breached' in pivot_table_yesterday.columns:
                        pivot_table_yesterday['SLA Breached %'] = (
                                (pivot_table_yesterday['SLA breached'] / pivot_table_yesterday['No. of orders']) * 100
                        ).fillna(0).map(lambda x: f"{x:.2f}%")
                    else:
                        pivot_table_yesterday['SLA Breached %'] = "0.00%"

                    # Add a total row
                    total_row_yesterday = pivot_table_yesterday.sum(numeric_only=True)
                    total_row_yesterday.name = 'Total'
                    pivot_table_yesterday = pd.concat([pivot_table_yesterday, total_row_yesterday.to_frame().T])

                    # Save the pivot table to CSV
                    filename_yesterday = f"Yesterday_{facility}_Export_Sale_Report_{datetime.now().strftime('%m%d%Y')}.csv"
                    pivot_table_yesterday.to_csv(filename_yesterday, index=True)

                    # Second Pivot Table (All Data)
                    pivot_table_all = facility_data.pivot_table(
                        values='SALE_ORDER_ITEM_STATUS',
                        index='CHANNEL_NAME',
                        columns='SLA',
                        aggfunc='count',
                        fill_value=0
                    )

                    # Ensure all required SLA columns are present
                    for col in required_sla_columns[:-2]:  # Exclude 'No. of orders' and 'SLA Breached %' from this loop
                        if col not in pivot_table_all.columns:
                            pivot_table_all[col] = 0

                    # Add the 'No. of orders' column
                    pivot_table_all['No. of orders'] = pivot_table_all.sum(axis=1)

                    # Add the 'SLA Breached %' column
                    if 'SLA breached' in pivot_table_all.columns:
                        pivot_table_all['SLA Breached %'] = (
                                (pivot_table_all['SLA breached'] / pivot_table_all['No. of orders']) * 100
                        ).fillna(0).map(lambda x: f"{x:.2f}%")
                    else:
                        pivot_table_all['SLA Breached %'] = "0.00%"

                    # Add a total row
                    total_row_all = pivot_table_all.sum(numeric_only=True)
                    total_row_all.name = 'Total'
                    pivot_table_all = pd.concat([pivot_table_all, total_row_all.to_frame().T])

                    # Save the pivot table to CSV
                    filename_all = f"{facility}_Export_Sale_Report_{datetime.now().strftime('%m%d%Y')}.csv"
                    pivot_table_all.to_csv(filename_all, index=True)

                    # Store both pivot tables
                    pivot_tables[f"{facility}_yesterday"] = pivot_table_yesterday
                    pivot_tables[f"{facility}_all"] = pivot_table_all

                # Render the template with all pivot tables
                return render_template(
                    'unicom.html',
                    pivot_tables=pivot_tables
                )

            else:
                return "Invalid file format. Allowed formats are CSV and Excel.", 400

        # Render the initial upload page for GET requests
        return render_template('upload.html')

    except Exception as e:
        # Handle unexpected errors
        return f"An error occurred: {str(e)}", 500


@app.route('/download_page/<filename>', methods=['GET'])
def download_file(filename):
    try:
        # Absolute path to the folder where the files are stored
        folder_path = os.path.abspath('exported_pivots')

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, filename)

        # Debugging line to confirm the path
        print(f"Attempting to download from: {file_path}")

        # Check if the file exists and send it for download
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"An error occurred while downloading the file: {str(e)}", 500


# Dictionary to hold uploaded file data
uploaded_files = {}

# Allowed file types
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/sla_report/eshopbox', methods=['GET', 'POST'])
def upload_sla_file():
    if request.method == 'POST':
        file = request.files.get("file")

        if not file or file.filename == '':
            flash("No file selected or file name is empty.")
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_data = file.read()
            uploaded_files[filename] = file_data
            flash(f"File '{filename}' uploaded successfully!")
            return redirect(url_for('eshopbox'))  # Redirect after upload to view or generate report
        else:
            flash("Invalid file type. Please upload a CSV or Excel file.")
            return redirect(request.url)

    return render_template('upload_eshopbox.html')


@app.route('/sla_report/eshopbox')
def eshopbox():
    # Display all uploaded files for the user to choose to view or generate SLA report
    return render_template('eshopbox.html', uploaded_files=list(uploaded_files.keys()))


# Simulated uploaded files dictionary
# uploaded_files = {
#     "file1.csv": b"Your file content here",
#     "file2.xlsx": b"Your file content here"
# }


@app.route('/sla_report/generate_sla_report', methods=['POST'])
def generate_sla_report():
    uploaded_data = {}  # Dictionary to store pivot table HTML for each file

    # Example: Simulate uploaded files (In real case, handle `request.files`)
    uploaded_files = request.files.to_dict(flat=False)

    for filename, file_data in uploaded_files.items():
        file_stream = io.BytesIO(file_data[0].read())

        # Load the file into a DataFrame
        if filename.endswith('.csv'):
            df = pd.read_csv(file_stream, low_memory=False)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_stream, low_memory=False)
        else:
            continue

        # Clean column names
        df.columns = df.columns.str.upper().str.replace(" ", "_")

        # Check if required columns exist
        required_columns = [
            'SHIPMENT_CREATED_IN_FLEX', 'EXPECTED_RTS_AT', 'PACKED_AT',
            'SHIPMENT_STATUS', 'SALES_CHANNEL', 'ORDER_ITEM_IDS'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Error: Missing required columns in {filename}: {', '.join(missing_columns)}"

        # Convert relevant columns to datetime
        for col in ['SHIPMENT_CREATED_IN_FLEX', 'EXPECTED_RTS_AT', 'PACKED_AT']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Add date columns
        df['EXPECTED_RTS_AT_DATE'] = df['EXPECTED_RTS_AT'].dt.date
        df['PACKED_AT_DATE'] = df['PACKED_AT'].dt.date

        # Handle missing Sales Channel values
        df['SALES_CHANNEL'] = df['SALES_CHANNEL'].fillna("Unknown").str.strip()

        # Exclude specific sales channels
        excluded_channels = ["CRED_FARMLEY_CMUM", "MEESHO_UNDERATED"]
        df_filtered = df[~df["SALES_CHANNEL"].isin(excluded_channels)]

        # Apply SLA logic
        df_filtered['SLA'] = df_filtered.apply(
            lambda x: "CANCELLED" if x['SHIPMENT_STATUS'] == "CANCELLED"
            else "Tech error" if x['SHIPMENT_STATUS'] in ["SIDELINED ON PACK", "SIDELINED ON HANDOVER"]
            else "With in SLA" if pd.notnull(x['PACKED_AT']) and x['PACKED_AT'] < x['EXPECTED_RTS_AT']
            else "SLA breached", axis=1
        )

        # Filter for yesterday's date
        yesterday_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize().date()
        filtered_df_yesterday = df_filtered[df_filtered['EXPECTED_RTS_AT_DATE'] == yesterday_date]

        # Ensure data exists before creating pivot
        if filtered_df_yesterday.empty:
            uploaded_data[filename] = f"No data available for {filename} on {yesterday_date}."
            continue

        # Create the pivot table
        pivot_table = filtered_df_yesterday.pivot_table(
            values='ORDER_ITEM_IDS',
            index='SALES_CHANNEL',
            columns='SLA',
            aggfunc='sum',
            fill_value=0
        )

        # Add Total and SLA Achieved % columns
        pivot_table['Total'] = pivot_table.sum(axis=1)
        if 'With in SLA' in pivot_table.columns:
            pivot_table['SLA Achieved %'] = (pivot_table['With in SLA'] / pivot_table['Total']) * 100
            pivot_table['SLA Achieved %'] = pivot_table['SLA Achieved %'].astype(int)
        else:
            pivot_table['SLA Achieved %'] = 0

        # Ensure all columns are included even if empty
        for column in ['CANCELLED', 'SLA breached', 'Tech error', 'With in SLA']:
            if column not in pivot_table.columns:
                pivot_table[column] = 0

        # Rearrange columns in a specific order
        column_order = ['CANCELLED', 'SLA breached', 'Tech error', 'With in SLA', 'Total', 'SLA Achieved %']
        pivot_table = pivot_table[column_order]

        # Convert the pivot table to HTML for display
        uploaded_data[filename] = pivot_table.to_html(
            classes='table table-striped', border=0, index=True
        )

    # Render the template and pass the data
    return render_template('sla_report.html', uploaded_data=uploaded_data)


@app.route('/sla_report/view_uploaded_data', methods=['GET'])
def view_uploaded_data():
    uploaded_data = {}
    for filename, file_data in uploaded_files.items():
        file_stream = io.BytesIO(file_data)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_stream)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_stream)
        else:
            continue
        uploaded_data[filename] = df.to_html(index=False)
    return render_template('view_uploaded_data.html', uploaded_data=uploaded_data)


# # Path to the directory where files are uploaded
# UPLOAD_FOLDER = 'path_to_your_upload_folder'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/sla_report/remove_upload_file', methods=['GET', 'POST'])
def remove_upload_file():
    try:
        # List all files in the upload folder
        files = os.listdir(app.config['UPLOAD_FOLDER'])

        # Sort files by creation time to get the most recent one
        files = sorted(files, key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)),
                       reverse=True)

        if not files:
            flash("No files found to remove.")
        else:
            # Get the most recent file
            recent_file = files[0]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], recent_file)

            # Remove the file
            os.remove(file_path)
            flash(f"File '{recent_file}' has been removed successfully.")
    except Exception as e:
        flash(f"An error occurred while removing the file: {e}")

    return render_template('remove_upload_file.html')


# pendency Report Route

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to process the uploaded file and generate a report
import datetime  # Ensure datetime module is properly imported


def process_file(uploaded_file, location_name):
    try:
        # Check if the file is empty
        if uploaded_file.readable():
            uploaded_file.seek(0)  # Reset the file pointer to the start of the file
            if not uploaded_file.read(1):  # Read one byte to check if the file is empty
                return {'error': f"Error processing file: The file for {location_name} is empty."}
            uploaded_file.seek(0)  # Reset again after checking

        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
        if df.empty:
            return {'error': f"Error processing file: No data found in the file for {location_name}."}

        # Standardize column names
        df.columns = df.columns.str.upper()
        df.columns = df.columns.str.replace(" ", "_")

        # Check for the required columns
        required_columns = ['SALES_CHANNEL', 'SHIPMENT_STATUS', 'SHIPMENT_ID', 'EXPECTED_RTS_AT',
                            'ORDER_CREATED_IN_ESHOPBOX', 'ORDER_ITEM_IDS', 'PACKED_AT']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                'error': f"Error processing file: Missing required columns {', '.join(missing_columns)} for {location_name}."}

        # Convert date columns to datetime
        for col in ['EXPECTED_RTS_AT', 'ORDER_CREATED_IN_ESHOPBOX', 'PACKED_AT']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert ORDER_ITEM_IDS to numeric
        if 'ORDER_ITEM_IDS' in df.columns:
            df['ORDER_ITEM_IDS'] = pd.to_numeric(df['ORDER_ITEM_IDS'], errors='coerce')

        # Filter the data based on the current date and necessary conditions
        today = datetime.datetime.now()  # Correct usage of datetime

        # Filter based on EXPECTED_RTS_AT
        df = df[df['EXPECTED_RTS_AT'] <= today]

        # Special handling for some sales channels
        special_channels = ['MEESHO_UNDERATED', 'CRED_FARMLEY_CMUM']
        mask_special = (df['SALES_CHANNEL'].isin(special_channels)) & (
                df['ORDER_CREATED_IN_ESHOPBOX'] < today.replace(hour=11, minute=59))
        four_pm_today = today.replace(hour=16, minute=0)
        df_special_filtered = df[mask_special & (df['EXPECTED_RTS_AT'] < four_pm_today)]
        df_other_filtered = df[~df['SALES_CHANNEL'].isin(special_channels)]

        # SLA Breach Handling
        sla_breach_filter = (df_other_filtered['EXPECTED_RTS_AT'].dt.date <= today.date()) & (
                df_other_filtered['SHIPMENT_STATUS'] != 'COMPLETED')
        df_sla_breached = df_other_filtered[sla_breach_filter]

        # Combine the filtered data
        filtered_df = pd.concat([df_special_filtered, df_other_filtered])
        excluded_statuses = ['CANCELLED', 'DISPATCHED', 'HANDOVER CANCELLED', 'HOLD', 'SIDELINED ON HANDOVER']
        filtered_df = filtered_df[~filtered_df['SHIPMENT_STATUS'].isin(excluded_statuses)]

        # Create a pivot table based on the filtered data
        pivot_table = filtered_df.pivot_table(
            index='SALES_CHANNEL',
            columns='SHIPMENT_STATUS',
            values='ORDER_ITEM_IDS',
            aggfunc='sum',
            fill_value=0,
            margins=True,  # This adds the 'Grand Total' row and column
            margins_name='Grand Total'  # Name of the total row/column
        )

        # --- SLA Logic (Yesterday) ---
        yesterday = (datetime.datetime.now() - timedelta(days=1)).date()
        df['EXPECTED_RTS_AT_DATE'] = df['EXPECTED_RTS_AT'].dt.date
        df1_yesterday = df[df['EXPECTED_RTS_AT_DATE'] == yesterday]

        # Apply the SLA logic
        df1_yesterday['SLA'] = df1_yesterday.apply(
            lambda x: "CANCELLED" if x['SHIPMENT_STATUS'] == "CANCELLED"
            else "Tech error" if x['SHIPMENT_STATUS'] in ["SIDELINED ON PACK", "SIDELINED ON HANDOVER"]
            else "With in SLA" if x['PACKED_AT'] < x['EXPECTED_RTS_AT']
            else "SLA breached", axis=1)

        # Create pivot table for SLA report
        pivot_table1 = df1_yesterday.pivot_table(values='ORDER_ITEM_IDS', index='SALES_CHANNEL', columns='SLA',
                                                 aggfunc='sum', fill_value=0)
        pivot_table1['Total'] = pivot_table1.sum(axis=1)
        sla_achieved = pivot_table1.get('With in SLA', 0)
        pivot_table1['SLA Achieved %'] = (sla_achieved / pivot_table1['Total']) * 100
        pivot_table1['SLA Achieved %'] = pivot_table1['SLA Achieved %'].fillna(0)

        # Return both pivot tables as a dictionary
        return {
            'pivot_table': pivot_table.to_html(classes='table table-bordered table-striped', index=True),
            'pivot_table1': pivot_table1.to_html(classes='table table-bordered table-striped', index=True)
        }

    except Exception as e:
        return {'error': f"Error processing file: {str(e)}"}


@app.route('/sla_report/pendency_sla', methods=['GET', 'POST'])
def pendency_sla():
    if request.method == 'POST':
        # Retrieve uploaded files
        uploaded_file_loc1 = request.files.get("uploaded_file_loc1")
        uploaded_file_loc2 = request.files.get("uploaded_file_loc2")
        uploaded_file_loc3 = request.files.get("uploaded_file_loc3")

        # Dictionary to store processed data for each location
        report_data = {}

        # Process files for each location
        if uploaded_file_loc1 and allowed_file(uploaded_file_loc1.filename):
            uploaded_file_loc1.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_loc1.filename))
            result = process_file(uploaded_file_loc1, "Gurgaon")
            if 'error' in result:
                flash(result['error'])
            else:
                report_data['Gurgaon'] = result

        if uploaded_file_loc2 and allowed_file(uploaded_file_loc2.filename):
            uploaded_file_loc2.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_loc2.filename))
            result = process_file(uploaded_file_loc2, "Mumbai")
            if 'error' in result:
                flash(result['error'])
            else:
                report_data['Mumbai'] = result

        if uploaded_file_loc3 and allowed_file(uploaded_file_loc3.filename):
            uploaded_file_loc3.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_loc3.filename))
            result = process_file(uploaded_file_loc3, "Hyderabad")
            if 'error' in result:
                flash(result['error'])
            else:
                report_data['Hyderabad'] = result

        # Check if any valid report data exists
        if report_data:
            return render_template('display_report.html', report_data=report_data)
        else:
            flash("No valid data to process. Please upload valid files.")
            return redirect(request.url)

    return render_template('pendency_sla.html')


@app.route('/show_report', methods=['GET'])
def show_report():
    return render_template('display_report.html')


@app.route('/view_data_location_wise', methods=['GET'])
def view_data_location_wise():
    # This route will render the data for each location-wise (you can customize this as needed)
    return render_template('view_data_location_wise.html')


# # Send email functionality (optional)
# def send_email(output, receiver_email):
#     message = MIMEMultipart()
#     message['Subject'] = "Pivot Table Report"
#     message['From'] = 'your_email@example.com'
#     message['To'] = receiver_email
#
#     body = MIMEText("Please find attached the pivot table report.", 'plain')
#     message.attach(body)
#
#     excel_attachment = MIMEApplication(output.getvalue(),
#                                        _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
#     excel_attachment.add_header('Content-Disposition', 'attachment', filename="pivot_table_report.xlsx")
#     message.attach(excel_attachment)
#
#     with smtplib.SMTP('smtp.office365.com', 587) as smtp_server:
#         smtp_server.starttls()
#         smtp_server.login('your_email@example.com', 'your_password')  # Use your credentials
#         smtp_server.sendmail('your_email@example.com', receiver_email, message.as_string())
#
#


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
@app.route('/download_dashboard')
def download_dashboard():
    # Load entries and convert to a DataFrame
    entries = load_entries()
    df = pd.DataFrame(entries)
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Entries')

    output.seek(0)
    return send_file(output, as_attachment=True, download_name='dashboard_data.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, render_template_string, session, \
    jsonify
import datetime
from datetime import datetime, timedelta
from datetime import datetime
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
import pyqrcode
import random
import os
import io
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


# Dashboard Route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# API for Graph Data
# API for Graph Data
@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    df = load_entries()

    if df.empty:
        return jsonify({"error": "No data available"})

    # Data Aggregation for Charts
    location_counts = df["Location"].value_counts().to_dict()
    customer_counts = df["Customer"].value_counts().to_dict()

    shift_counts = df["Shift"].value_counts().to_dict()
    inward_counts = df["Inward Name_opening"].value_counts().to_dict()

    # Response Data
    return jsonify({
        "locations": location_counts,
        "customers": customer_counts,
        "shifts": shift_counts,
        "inwards": inward_counts
    })


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
        form_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Opening data ko opening.csv me store karein
        save_entry_to_csv(form_data, OPENING_CSV)

        return render_template('opening.html', message="Opening data submitted successfully!")

    return render_template('opening.html')

@app.route('/dwm_report/closing', methods=['GET', 'POST'])
def closing():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['source'] = 'Closing'
        form_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        return pd.DataFrame()  # Agar koi file nahi mili to empty DataFrame return karein
        # Ensure Date column is in datetime format
    df_opening["Date"] = pd.to_datetime(df_opening["Date"], errors='coerce')
    df_closing["Date"] = pd.to_datetime(df_closing["Date"], errors='coerce')
    # Suffix lagane ke liye excluded columns
    excluded_columns = {"Date", "Shift", "Location", "Customer"}

    # Opening aur Closing columns ke suffixes
    df_opening = df_opening.rename(
        columns={col: f"{col}_opening" if col not in excluded_columns else col for col in df_opening.columns}
    )
    df_closing = df_closing.rename(
        columns={col: f"{col}_closing" if col not in excluded_columns else col for col in df_closing.columns}
    )

    # Merge on Date and Shift
    merged_df = pd.merge(df_opening, df_closing, on=["Date", "Shift", "Location", "Customer"], how="outer")

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
        {"Department": "Inventory", "Activity": "Qty- Good Putaway", "Benchmark": 1500},
        {"Department": "Inventory", "Activity": "Qty- Cycle Count/Consolidation", "Benchmark": 2000},
        {"Department": "Inventory", "Activity": "STN-Direct Putaway", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Qty. picked-B2C", "Benchmark": 400},
        {"Department": "Outward", "Activity": "QTY(Invoiced+Packed)B2C", "Benchmark": 450},
        {"Department": "Outward", "Activity": "QTY.Manifest+Handover-B2C", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Picked QTY. ( B2B )", "Benchmark": 600},
        {"Department": "Outward", "Activity": "QTY(Invoiced+Packed)B2B", "Benchmark": 800},
        {"Department": "Return", "Activity": "RTO Received QTY. (B2B/B2C)", "Benchmark": 1000},
        {"Department": "Return", "Activity": "RTO Good processing return", "Benchmark": 200},
        {"Department": "Return", "Activity": "Bad processing with claim", "Benchmark": 60},
        {"Department": "Return", "Activity": "RTO Putaway", "Benchmark": 1200},
        {"Department": "Return", "Activity": "Qty. GP Creation QCR", "Benchmark": 1000},
        {"Department": "Other Activities", "Activity": "", "Benchmark": 0},
    ]

    benchmarks_df = pd.DataFrame(benchmark_data)

    # Manpower and Execution Calculations
    benchmarks_df['Deployed Manpower'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity} Manpower_opening", pd.Series()).sum()
    )

    benchmarks_df['Target'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity}_opening", pd.Series()).sum()
    )

    # Required Manpower Calculation
    benchmarks_df['Required Manpower'] = (benchmarks_df['Target'] / benchmarks_df['Benchmark']).round(2)
    benchmarks_df['Extra Manpower'] = benchmarks_df['Deployed Manpower'] - benchmarks_df['Required Manpower'].round(2)

    benchmarks_df['Execution'] = benchmarks_df['Activity'].apply(
        lambda activity: df.get(f"{activity}_closing", pd.Series()).sum()
    )

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
        'Department', 'Activity', 'Benchmark', 'Head Count', 'Deployed Manpower',
        'Target', 'Required Manpower', 'Extra Manpower', 'Execution',
        'Capacity','Planned Load', 'Capacity Vs Execution', 'Target Vs Execution']
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


@app.route('/dwm_report/download_dashboard_ai', methods=['GET'])
def download_dashboard_ai():
    # Fetch the filtered data (same as passed to the template)
    table_data = session.get('table_data', None)

    if not table_data:
        return "No data available for download", 400

    # Convert the list of records (table data) back to a DataFrame
    df_for_download = pd.DataFrame(table_data)

    # Apply the same filters that were applied on the dashboard page
    selected_date = request.args.get('Date', default=None)
    selected_shift = request.args.get('Shift', default=None)
    selected_location = request.args.get('Location', default=None)
    selected_customer = request.args.get('Customer', default=None)

    # Apply filters to the DataFrame
    if selected_date:
        df_for_download = df_for_download[df_for_download["Date"] == selected_date]
    if selected_shift:
        df_for_download = df_for_download[df_for_download["Shift"] == selected_shift]
    if selected_location:
        df_for_download = df_for_download[df_for_download["Location"] == selected_location]
    if selected_customer:
        df_for_download = df_for_download[df_for_download["Customer"] == selected_customer]

    # Define the correct column order to match the dashboard display
    column_order = [
        'Department', 'Activity', 'Benchmark', 'Deployed Manpower',
        'Required Manpower', 'Extra Manpower', 'Target', 'Capacity',
        'Execution', 'Capacity Vs Execution', 'Target Vs Execution'
    ]

    # Reorder the columns in the DataFrame
    df_for_download = df_for_download[column_order]

    # Convert the filtered DataFrame to CSV format
    csv_data = df_for_download.to_csv(index=False)

    # Create an in-memory buffer for the CSV file
    buffer = io.BytesIO()
    buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)

    # Send the filtered CSV file as a response
    return send_file(buffer, as_attachment=True, download_name="filtered_dwm_dashboard_data.csv", mimetype="text/csv")


def calculate_required_manpower(value):
    """
    Example function to calculate required manpower.
    Adjust the logic based on actual requirements.
    """
    try:
        return int(value) * 2  # Example: Required manpower is twice the value
    except (ValueError, TypeError):
        return "N/A"


# Project Report Route
# Project Report Route


# Load initial warehouse data (Simulated function)
def load_entries():
    return session.get('table_data', [])


MASTER_FILE_PATH = os.path.join(UPLOAD_FOLDER, 'master_data.xlsx')


def load_master_data():
    if os.path.exists(MASTER_FILE_PATH):
        master_df = pd.read_excel(MASTER_FILE_PATH, engine='openpyxl')
        required_columns = {"Department", "Activity", "Benchmark", "Planned Load", "Head Count"}
        if required_columns.issubset(master_df.columns):
            return master_df
    return pd.DataFrame(columns=["Department", "Activity", "Benchmark", "Planned Load", "Head Count"])


@app.route('/upload_master', methods=['POST'])
def upload_master():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    file.save(MASTER_FILE_PATH)
    return redirect(url_for('projection_dashboard'))


@app.route('/dwm_report/projection_dashboard', methods=['GET'])
def projection_dashboard():
    table_data_list = session.get('table_data', [])
    df = pd.DataFrame(table_data_list)
    master_df = load_master_data()

    if not df.empty:
        df = df.merge(master_df, on="Activity", how="left", suffixes=("", "_master"))
        for col in ["Benchmark", "Planned Load", "Head Count"]:
            if col in df.columns and f"{col}_master" in df.columns:
                df[col] = df[f"{col}_master"].fillna(df[col])
                df.drop(columns=[f"{col}_master"], inplace=True)

    filters = {key: request.args.get(key) for key in ["Date", "Shift", "Location", "Customer"] if request.args.get(key)}
    for key, value in filters.items():
        df = df[df[key] == value]

    table_data_list = df.to_dict(orient='records')
    unique_filters = {col: df[col].dropna().unique().tolist() if col in df.columns else [] for col in
                      ["Date", "Shift", "Location", "Customer"]}

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Warehouse Projection Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 20px;
                    padding: 20px;
                    text-align: center;
                }
                h2 {
                    color: #333;
                }
                form {
                    margin-bottom: 20px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    background: #fff;
                }
                th, td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                th {
                    background-color: #007bff;
                    color: white;
                }
                select, button {
                    padding: 8px;
                    margin: 5px;
                    border-radius: 5px;
                }
                button {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <h2>Upload Master File</h2>
            <form action="/upload_master" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".xlsx">
                <button type="submit">Upload</button>
            </form>
            <h2>Filter Data</h2>
            <form method="GET">
                {% for key, values in unique_filters.items() %}
                <label>{{ key }}:</label>
                <select name="{{ key }}">
                    <option value="">--Select--</option>
                    {% for value in values %}
                    <option value="{{ value }}">{{ value }}</option>
                    {% endfor %}
                </select>
                {% endfor %}
                <button type="submit">Apply</button>
            </form>
            <h2>Dashboard Data</h2>
            {% if table_data_list %}
            <table>
                <tr>
                    {% for col in table_data_list[0].keys() %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
                {% for row in table_data_list %}
                <tr>
                    {% for value in row.values() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No data available.</p>
            {% endif %}
        </body>
        </html>
    """, table_data_list=table_data_list, unique_filters=unique_filters)


# @dash_app.server.route('/projection_dashboard')
# def projection_dashboard():
#     return render_template('projection_dashboard.html')


# SLA Report Route
@app.route('/sla_report')
def sla_report():
    return render_template('sla_report.html')


ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/sla_report/unicom', methods=['GET', 'POST'])
def unicom():
    try:
        if request.method == 'POST':
            # Check if the request contains a file
            if 'file' not in request.files:
                return "No file part in the request", 400

            file = request.files['file']

            # Check if a file is selected
            if file.filename == '':
                return "No file selected", 400

            # Validate and save the file
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Load the file into a DataFrame
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)
                else:
                    return "Invalid file format. Allowed formats are CSV and Excel.", 400

                # Ensure the dataframe is not empty
                if df.empty:
                    return "Uploaded file contains no data.", 400

                # Preprocess the dataframe
                df.columns = df.columns.str.upper().str.replace(" ", "_")
                df['FULFILLMENT_TAT'] = pd.to_datetime(df['FULFILLMENT_TAT'], errors='coerce')
                df['INVOICE_CREATED'] = pd.to_datetime(df['INVOICE_CREATED'], errors='coerce')

                # Check if required columns exist
                required_columns = ['FULFILLMENT_TAT', 'INVOICE_CREATED', 'SALE_ORDER_ITEM_STATUS', 'FACILITY',
                                    'CHANNEL_NAME']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    return f"Missing required columns: {', '.join(missing_columns)}", 400

                # Filter for yesterday's data
                yesterday_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize().date()
                df['FULFILLMENT_TAT_DATE'] = pd.to_datetime(df['FULFILLMENT_TAT'].dt.date, errors='coerce')
                df = df[df['FULFILLMENT_TAT_DATE'] <= pd.Timestamp(yesterday_date)]

                # Add SLA status
                df['SLA'] = df.apply(
                    lambda x: "Cancelled" if x['SALE_ORDER_ITEM_STATUS'] == "CANCELLED"
                    else "With in SLA" if x['INVOICE_CREATED'] <= x['FULFILLMENT_TAT']
                    else "SLA breached", axis=1)

                # Filter data for yesterday
                filtered_df = df[(
                                         df['FULFILLMENT_TAT_DATE'] == pd.Timestamp(yesterday_date)) &
                                 ((df['INVOICE_CREATED'] >= pd.Timestamp(yesterday_date)) | (
                                     df['INVOICE_CREATED'].isnull()))
                                 ]

                # Ensure there is data after filtering
                if filtered_df.empty:
                    return "No data available for the selected date range.", 400

                # Create pivot tables for specified facilities
                pivot_tables = {}
                facilities = ['Kothari_HYD', 'Kothari_GGN']

                required_sla_columns = ['Cancelled', 'SLA breached', 'With in SLA', 'No. of orders', 'SLA Breached %']

                for facility in facilities:
                    facility_data = filtered_df[filtered_df['FACILITY'] == facility]

                    # First Pivot Table (Yesterday's Data)
                    pivot_table_yesterday = facility_data.pivot_table(
                        values='SALE_ORDER_ITEM_STATUS',
                        index='CHANNEL_NAME',
                        columns='SLA',
                        aggfunc='count',
                        fill_value=0
                    )

                    # Ensure all required SLA columns are present
                    for col in required_sla_columns[:-2]:  # Exclude 'No. of orders' and 'SLA Breached %' from this loop
                        if col not in pivot_table_yesterday.columns:
                            pivot_table_yesterday[col] = 0

                    # Add the 'No. of orders' column
                    pivot_table_yesterday['No. of orders'] = pivot_table_yesterday.sum(axis=1)

                    # Add the 'SLA Breached %' column
                    if 'SLA breached' in pivot_table_yesterday.columns:
                        pivot_table_yesterday['SLA Breached %'] = (
                                (pivot_table_yesterday['SLA breached'] / pivot_table_yesterday['No. of orders']) * 100
                        ).fillna(0).map(lambda x: f"{x:.2f}%")
                    else:
                        pivot_table_yesterday['SLA Breached %'] = "0.00%"

                    # Add a total row
                    total_row_yesterday = pivot_table_yesterday.sum(numeric_only=True)
                    total_row_yesterday.name = 'Total'
                    pivot_table_yesterday = pd.concat([pivot_table_yesterday, total_row_yesterday.to_frame().T])

                    # Save the pivot table to CSV
                    filename_yesterday = f"Yesterday_{facility}_Export_Sale_Report_{datetime.now().strftime('%m%d%Y')}.csv"
                    pivot_table_yesterday.to_csv(filename_yesterday, index=True)

                    # Second Pivot Table (All Data)
                    pivot_table_all = facility_data.pivot_table(
                        values='SALE_ORDER_ITEM_STATUS',
                        index='CHANNEL_NAME',
                        columns='SLA',
                        aggfunc='count',
                        fill_value=0
                    )

                    # Ensure all required SLA columns are present
                    for col in required_sla_columns[:-2]:  # Exclude 'No. of orders' and 'SLA Breached %' from this loop
                        if col not in pivot_table_all.columns:
                            pivot_table_all[col] = 0

                    # Add the 'No. of orders' column
                    pivot_table_all['No. of orders'] = pivot_table_all.sum(axis=1)

                    # Add the 'SLA Breached %' column
                    if 'SLA breached' in pivot_table_all.columns:
                        pivot_table_all['SLA Breached %'] = (
                                (pivot_table_all['SLA breached'] / pivot_table_all['No. of orders']) * 100
                        ).fillna(0).map(lambda x: f"{x:.2f}%")
                    else:
                        pivot_table_all['SLA Breached %'] = "0.00%"

                    # Add a total row
                    total_row_all = pivot_table_all.sum(numeric_only=True)
                    total_row_all.name = 'Total'
                    pivot_table_all = pd.concat([pivot_table_all, total_row_all.to_frame().T])

                    # Save the pivot table to CSV
                    filename_all = f"{facility}_Export_Sale_Report_{datetime.now().strftime('%m%d%Y')}.csv"
                    pivot_table_all.to_csv(filename_all, index=True)

                    # Store both pivot tables
                    pivot_tables[f"{facility}_yesterday"] = pivot_table_yesterday
                    pivot_tables[f"{facility}_all"] = pivot_table_all

                # Render the template with all pivot tables
                return render_template(
                    'unicom.html',
                    pivot_tables=pivot_tables
                )

            else:
                return "Invalid file format. Allowed formats are CSV and Excel.", 400

        # Render the initial upload page for GET requests
        return render_template('upload.html')

    except Exception as e:
        # Handle unexpected errors
        return f"An error occurred: {str(e)}", 500


@app.route('/download_page/<filename>', methods=['GET'])
def download_file(filename):
    try:
        # Absolute path to the folder where the files are stored
        folder_path = os.path.abspath('exported_pivots')

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, filename)

        # Debugging line to confirm the path
        print(f"Attempting to download from: {file_path}")

        # Check if the file exists and send it for download
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"An error occurred while downloading the file: {str(e)}", 500


# Dictionary to hold uploaded file data
uploaded_files = {}

# Allowed file types
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/sla_report/eshopbox', methods=['GET', 'POST'])
def upload_sla_file():
    if request.method == 'POST':
        file = request.files.get("file")

        if not file or file.filename == '':
            flash("No file selected or file name is empty.")
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_data = file.read()
            uploaded_files[filename] = file_data
            flash(f"File '{filename}' uploaded successfully!")
            return redirect(url_for('eshopbox'))  # Redirect after upload to view or generate report
        else:
            flash("Invalid file type. Please upload a CSV or Excel file.")
            return redirect(request.url)

    return render_template('upload_eshopbox.html')


@app.route('/sla_report/eshopbox')
def eshopbox():
    # Display all uploaded files for the user to choose to view or generate SLA report
    return render_template('eshopbox.html', uploaded_files=list(uploaded_files.keys()))


# Simulated uploaded files dictionary
# uploaded_files = {
#     "file1.csv": b"Your file content here",
#     "file2.xlsx": b"Your file content here"
# }


@app.route('/sla_report/generate_sla_report', methods=['POST'])
def generate_sla_report():
    uploaded_data = {}  # Dictionary to store pivot table HTML for each file

    # Example: Simulate uploaded files (In real case, handle `request.files`)
    uploaded_files = request.files.to_dict(flat=False)

    for filename, file_data in uploaded_files.items():
        file_stream = io.BytesIO(file_data[0].read())

        # Load the file into a DataFrame
        if filename.endswith('.csv'):
            df = pd.read_csv(file_stream, low_memory=False)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_stream, low_memory=False)
        else:
            continue

        # Clean column names
        df.columns = df.columns.str.upper().str.replace(" ", "_")

        # Check if required columns exist
        required_columns = [
            'SHIPMENT_CREATED_IN_FLEX', 'EXPECTED_RTS_AT', 'PACKED_AT',
            'SHIPMENT_STATUS', 'SALES_CHANNEL', 'ORDER_ITEM_IDS'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Error: Missing required columns in {filename}: {', '.join(missing_columns)}"

        # Convert relevant columns to datetime
        for col in ['SHIPMENT_CREATED_IN_FLEX', 'EXPECTED_RTS_AT', 'PACKED_AT']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Add date columns
        df['EXPECTED_RTS_AT_DATE'] = df['EXPECTED_RTS_AT'].dt.date
        df['PACKED_AT_DATE'] = df['PACKED_AT'].dt.date

        # Handle missing Sales Channel values
        df['SALES_CHANNEL'] = df['SALES_CHANNEL'].fillna("Unknown").str.strip()

        # Exclude specific sales channels
        excluded_channels = ["CRED_FARMLEY_CMUM", "MEESHO_UNDERATED"]
        df_filtered = df[~df["SALES_CHANNEL"].isin(excluded_channels)]

        # Apply SLA logic
        df_filtered['SLA'] = df_filtered.apply(
            lambda x: "CANCELLED" if x['SHIPMENT_STATUS'] == "CANCELLED"
            else "Tech error" if x['SHIPMENT_STATUS'] in ["SIDELINED ON PACK", "SIDELINED ON HANDOVER"]
            else "With in SLA" if pd.notnull(x['PACKED_AT']) and x['PACKED_AT'] < x['EXPECTED_RTS_AT']
            else "SLA breached", axis=1
        )

        # Filter for yesterday's date
        yesterday_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize().date()
        filtered_df_yesterday = df_filtered[df_filtered['EXPECTED_RTS_AT_DATE'] == yesterday_date]

        # Ensure data exists before creating pivot
        if filtered_df_yesterday.empty:
            uploaded_data[filename] = f"No data available for {filename} on {yesterday_date}."
            continue

        # Create the pivot table
        pivot_table = filtered_df_yesterday.pivot_table(
            values='ORDER_ITEM_IDS',
            index='SALES_CHANNEL',
            columns='SLA',
            aggfunc='sum',
            fill_value=0
        )

        # Add Total and SLA Achieved % columns
        pivot_table['Total'] = pivot_table.sum(axis=1)
        if 'With in SLA' in pivot_table.columns:
            pivot_table['SLA Achieved %'] = (pivot_table['With in SLA'] / pivot_table['Total']) * 100
            pivot_table['SLA Achieved %'] = pivot_table['SLA Achieved %'].astype(int)
        else:
            pivot_table['SLA Achieved %'] = 0

        # Ensure all columns are included even if empty
        for column in ['CANCELLED', 'SLA breached', 'Tech error', 'With in SLA']:
            if column not in pivot_table.columns:
                pivot_table[column] = 0

        # Rearrange columns in a specific order
        column_order = ['CANCELLED', 'SLA breached', 'Tech error', 'With in SLA', 'Total', 'SLA Achieved %']
        pivot_table = pivot_table[column_order]

        # Convert the pivot table to HTML for display
        uploaded_data[filename] = pivot_table.to_html(
            classes='table table-striped', border=0, index=True
        )

    # Render the template and pass the data
    return render_template('sla_report.html', uploaded_data=uploaded_data)


@app.route('/sla_report/view_uploaded_data', methods=['GET'])
def view_uploaded_data():
    uploaded_data = {}
    for filename, file_data in uploaded_files.items():
        file_stream = io.BytesIO(file_data)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_stream)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_stream)
        else:
            continue
        uploaded_data[filename] = df.to_html(index=False)
    return render_template('view_uploaded_data.html', uploaded_data=uploaded_data)


# # Path to the directory where files are uploaded
# UPLOAD_FOLDER = 'path_to_your_upload_folder'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/sla_report/remove_upload_file', methods=['GET', 'POST'])
def remove_upload_file():
    try:
        # List all files in the upload folder
        files = os.listdir(app.config['UPLOAD_FOLDER'])

        # Sort files by creation time to get the most recent one
        files = sorted(files, key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)),
                       reverse=True)

        if not files:
            flash("No files found to remove.")
        else:
            # Get the most recent file
            recent_file = files[0]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], recent_file)

            # Remove the file
            os.remove(file_path)
            flash(f"File '{recent_file}' has been removed successfully.")
    except Exception as e:
        flash(f"An error occurred while removing the file: {e}")

    return render_template('remove_upload_file.html')


# pendency Report Route

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to process the uploaded file and generate a report
import datetime  # Ensure datetime module is properly imported


def process_file(uploaded_file, location_name):
    try:
        # Check if the file is empty
        if uploaded_file.readable():
            uploaded_file.seek(0)  # Reset the file pointer to the start of the file
            if not uploaded_file.read(1):  # Read one byte to check if the file is empty
                return {'error': f"Error processing file: The file for {location_name} is empty."}
            uploaded_file.seek(0)  # Reset again after checking

        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
        if df.empty:
            return {'error': f"Error processing file: No data found in the file for {location_name}."}

        # Standardize column names
        df.columns = df.columns.str.upper()
        df.columns = df.columns.str.replace(" ", "_")

        # Check for the required columns
        required_columns = ['SALES_CHANNEL', 'SHIPMENT_STATUS', 'SHIPMENT_ID', 'EXPECTED_RTS_AT',
                            'ORDER_CREATED_IN_ESHOPBOX', 'ORDER_ITEM_IDS', 'PACKED_AT']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                'error': f"Error processing file: Missing required columns {', '.join(missing_columns)} for {location_name}."}

        # Convert date columns to datetime
        for col in ['EXPECTED_RTS_AT', 'ORDER_CREATED_IN_ESHOPBOX', 'PACKED_AT']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert ORDER_ITEM_IDS to numeric
        if 'ORDER_ITEM_IDS' in df.columns:
            df['ORDER_ITEM_IDS'] = pd.to_numeric(df['ORDER_ITEM_IDS'], errors='coerce')

        # Filter the data based on the current date and necessary conditions
        today = datetime.datetime.now()  # Correct usage of datetime

        # Filter based on EXPECTED_RTS_AT
        df = df[df['EXPECTED_RTS_AT'] <= today]

        # Special handling for some sales channels
        special_channels = ['MEESHO_UNDERATED', 'CRED_FARMLEY_CMUM']
        mask_special = (df['SALES_CHANNEL'].isin(special_channels)) & (
                df['ORDER_CREATED_IN_ESHOPBOX'] < today.replace(hour=11, minute=59))
        four_pm_today = today.replace(hour=16, minute=0)
        df_special_filtered = df[mask_special & (df['EXPECTED_RTS_AT'] < four_pm_today)]
        df_other_filtered = df[~df['SALES_CHANNEL'].isin(special_channels)]

        # SLA Breach Handling
        sla_breach_filter = (df_other_filtered['EXPECTED_RTS_AT'].dt.date <= today.date()) & (
                df_other_filtered['SHIPMENT_STATUS'] != 'COMPLETED')
        df_sla_breached = df_other_filtered[sla_breach_filter]

        # Combine the filtered data
        filtered_df = pd.concat([df_special_filtered, df_other_filtered])
        excluded_statuses = ['CANCELLED', 'DISPATCHED', 'HANDOVER CANCELLED', 'HOLD', 'SIDELINED ON HANDOVER']
        filtered_df = filtered_df[~filtered_df['SHIPMENT_STATUS'].isin(excluded_statuses)]

        # Create a pivot table based on the filtered data
        pivot_table = filtered_df.pivot_table(
            index='SALES_CHANNEL',
            columns='SHIPMENT_STATUS',
            values='ORDER_ITEM_IDS',
            aggfunc='sum',
            fill_value=0,
            margins=True,  # This adds the 'Grand Total' row and column
            margins_name='Grand Total'  # Name of the total row/column
        )

        # --- SLA Logic (Yesterday) ---
        yesterday = (datetime.datetime.now() - timedelta(days=1)).date()
        df['EXPECTED_RTS_AT_DATE'] = df['EXPECTED_RTS_AT'].dt.date
        df1_yesterday = df[df['EXPECTED_RTS_AT_DATE'] == yesterday]

        # Apply the SLA logic
        df1_yesterday['SLA'] = df1_yesterday.apply(
            lambda x: "CANCELLED" if x['SHIPMENT_STATUS'] == "CANCELLED"
            else "Tech error" if x['SHIPMENT_STATUS'] in ["SIDELINED ON PACK", "SIDELINED ON HANDOVER"]
            else "With in SLA" if x['PACKED_AT'] < x['EXPECTED_RTS_AT']
            else "SLA breached", axis=1)

        # Create pivot table for SLA report
        pivot_table1 = df1_yesterday.pivot_table(values='ORDER_ITEM_IDS', index='SALES_CHANNEL', columns='SLA',
                                                 aggfunc='sum', fill_value=0)
        pivot_table1['Total'] = pivot_table1.sum(axis=1)
        sla_achieved = pivot_table1.get('With in SLA', 0)
        pivot_table1['SLA Achieved %'] = (sla_achieved / pivot_table1['Total']) * 100
        pivot_table1['SLA Achieved %'] = pivot_table1['SLA Achieved %'].fillna(0)

        # Return both pivot tables as a dictionary
        return {
            'pivot_table': pivot_table.to_html(classes='table table-bordered table-striped', index=True),
            'pivot_table1': pivot_table1.to_html(classes='table table-bordered table-striped', index=True)
        }

    except Exception as e:
        return {'error': f"Error processing file: {str(e)}"}


@app.route('/sla_report/pendency_sla', methods=['GET', 'POST'])
def pendency_sla():
    if request.method == 'POST':
        # Retrieve uploaded files
        uploaded_file_loc1 = request.files.get("uploaded_file_loc1")
        uploaded_file_loc2 = request.files.get("uploaded_file_loc2")
        uploaded_file_loc3 = request.files.get("uploaded_file_loc3")

        # Dictionary to store processed data for each location
        report_data = {}

        # Process files for each location
        if uploaded_file_loc1 and allowed_file(uploaded_file_loc1.filename):
            uploaded_file_loc1.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_loc1.filename))
            result = process_file(uploaded_file_loc1, "Gurgaon")
            if 'error' in result:
                flash(result['error'])
            else:
                report_data['Gurgaon'] = result

        if uploaded_file_loc2 and allowed_file(uploaded_file_loc2.filename):
            uploaded_file_loc2.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_loc2.filename))
            result = process_file(uploaded_file_loc2, "Mumbai")
            if 'error' in result:
                flash(result['error'])
            else:
                report_data['Mumbai'] = result

        if uploaded_file_loc3 and allowed_file(uploaded_file_loc3.filename):
            uploaded_file_loc3.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_loc3.filename))
            result = process_file(uploaded_file_loc3, "Hyderabad")
            if 'error' in result:
                flash(result['error'])
            else:
                report_data['Hyderabad'] = result

        # Check if any valid report data exists
        if report_data:
            return render_template('display_report.html', report_data=report_data)
        else:
            flash("No valid data to process. Please upload valid files.")
            return redirect(request.url)

    return render_template('pendency_sla.html')


@app.route('/show_report', methods=['GET'])
def show_report():
    return render_template('display_report.html')


@app.route('/view_data_location_wise', methods=['GET'])
def view_data_location_wise():
    # This route will render the data for each location-wise (you can customize this as needed)
    return render_template('view_data_location_wise.html')


# # Send email functionality (optional)
# def send_email(output, receiver_email):
#     message = MIMEMultipart()
#     message['Subject'] = "Pivot Table Report"
#     message['From'] = 'your_email@example.com'
#     message['To'] = receiver_email
#
#     body = MIMEText("Please find attached the pivot table report.", 'plain')
#     message.attach(body)
#
#     excel_attachment = MIMEApplication(output.getvalue(),
#                                        _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
#     excel_attachment.add_header('Content-Disposition', 'attachment', filename="pivot_table_report.xlsx")
#     message.attach(excel_attachment)
#
#     with smtplib.SMTP('smtp.office365.com', 587) as smtp_server:
#         smtp_server.starttls()
#         smtp_server.login('your_email@example.com', 'your_password')  # Use your credentials
#         smtp_server.sendmail('your_email@example.com', receiver_email, message.as_string())
#
#


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
@app.route('/download_dashboard')
def download_dashboard():
    # Load entries and convert to a DataFrame
    entries = load_entries()
    df = pd.DataFrame(entries)
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Entries')

    output.seek(0)
    return send_file(output, as_attachment=True, download_name='dashboard_data.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 2100dd6 (Fixed master_file merge issue)
