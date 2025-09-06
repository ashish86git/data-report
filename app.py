from flask import Flask, render_template, request, redirect, url_for, send_file, flash, render_template_string, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor
from functools import wraps
import pandas as pd
import numpy as np
from io import BytesIO
import pyqrcode
import random
import os
import io
import psycopg2
from urllib.parse import urlparse
import smtplib
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import zipfile
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret_key"

# ✅ Dummy user store (future me DB use kar sakte ho)
users = {}

# ✅ Heroku PostgreSQL Connection Setup
DATABASE_URL = "postgres://u7tqojjihbpn7s:p1b1897f6356bab4e52b727ee100290a84e4bf71d02e064e90c2c705bfd26f4a5@c7s7ncbk19n97r.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d8lp4hr6fmvb9m"
url = urlparse(DATABASE_URL)

def get_db_connection():
    return psycopg2.connect(
        database=url.path[1:],   # database name
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )

conn = get_db_connection()
cur = conn.cursor()
# Define the path to the file for storing entries
entries_file = 'entries.csv'

# Folder to store uploaded files temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])



def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:   # agar session me user nahi hai to
            flash("Please login first!", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_entries():
    if os.path.exists(entries_file):
        return pd.read_csv(entries_file).to_dict(orient='records')
    return []


def save_entries(entries):
    df = pd.DataFrame(entries)
    df.to_csv(entries_file, index=False)


# ------------------ 🔑 AUTH SYSTEM ------------------

# ===================== 🔑 AUTH SYSTEM =====================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        role = request.form["role"]
        location = request.form["location"]   # ✅ new
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("❌ Passwords do not match!", "danger")
            return redirect(url_for("register"))

        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # ⚠️ Plain password save (as per your request)
            cur.execute(
                "INSERT INTO user_dwm (email, password, role, location) VALUES (%s, %s, %s, %s)",
                (email, password, role, location)   # ✅ location added
            )

            conn.commit()
            cur.close()
            conn.close()

            flash("✅ Registration successful! Please login.", "success")
            return redirect(url_for("login"))

        except psycopg2.Error as e:
            flash(f"⚠️ Error: {e.pgerror}", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT * FROM user_dwm WHERE email = %s", (email,))
        user = cur.fetchone()

        cur.close()
        conn.close()

        # ⚠️ Plain password comparison (as-is)
        if user and user['password'] == password:
            session['user'] = user['email']
            session['role'] = user['role']
            session['location'] = user.get('location')   # ✅ NEW
            flash("✅ Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("❌ Invalid credentials!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')




@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("role", None)
    session.pop("location", None)   # ✅ clear location
    flash("✅ Logged out successfully.", "success")
    return redirect(url_for("login"))



# ------------------ 🔑 END AUTH ------------------


# Home Route (🔒 Protected)
@app.route('/')
@login_required
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/dwm_report')
@login_required
def dwm_report():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dwm_report.html')


OPENING_CSV = "opening.csv"
CLOSING_CSV = "closing.csv"


def save_entry_to_csv(entry, filename):
    df = pd.DataFrame([entry])
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))


# ------------------ Opening Form ------------------
@app.route('/dwm_report/opening', methods=['GET', 'POST'])
@login_required
def opening():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        integer_fields = [
            'grn_qty_pendency', 'stn_qty_pendency', 'putaway_cancel_qty_pendency',
            'putaway_return_qty_pendency', 'grn_sellable_qty_pendency', 'bin_movement_pendency',
            'return_pendency', 'rtv_pendency', 'channel_order_qty_b2c_pendency',
            'rts_order_qty_b2c_pendency', 'breached_qty_pendency', 'side_lined_pendency',
            'dispatch_not_marked', 'not_dispatched_orders', 'no_of_floor_associated',
            'unloading_loading_boxes', 'receipt_process_boxes', 'qty_grn_qc',
            'qty_good_putaway', 'qty_cycle_count', 'stn_direct_putaway',
            'qty_picked_b2c', 'qty_invoiced_packed_b2c', 'qty_manifest_handover_b2c',
            'qty_invoiced_packed_b2b', 'picked_qty_b2b', 'rto_received_qty',
            'rto_putaway_qty', 'qty_gp_creation_qcr', 'rto_good_processing_return',
            'bad_processing_with_claim'
        ]

        decimal_fields = [
            'unloading_loading_boxes_manpower', 'receipt_process_boxes_manpower',
            'qty_grn_qc_manpower', 'qty_good_putaway_manpower', 'qty_cycle_count_manpower',
            'stn_direct_putaway_manpower', 'qty_picked_b2c_manpower',
            'qty_invoiced_packed_b2c_manpower', 'qty_manifest_handover_b2c_manpower',
            'qty_invoiced_packed_b2b_manpower', 'picked_qty_b2b_manpower',
            'rto_received_qty_manpower', 'rto_putaway_qty_manpower',
            'qty_gp_creation_qcr_manpower', 'rto_good_processing_return_manpower',
            'bad_processing_with_claim_manpower'
        ]

        cleaned_data = {}
        for key in request.form:
            value = request.form[key].strip()
            if key in integer_fields:
                try:
                    cleaned_data[key] = int(value) if value != '' else None
                except ValueError:
                    cleaned_data[key] = None
            elif key in decimal_fields:
                try:
                    cleaned_data[key] = Decimal(value) if value != '' else None
                except (ValueError, InvalidOperation):
                    cleaned_data[key] = None
            else:
                cleaned_data[key] = value if value != '' else None

        timestamp_now = datetime.now()
        cleaned_data['source'] = 'Opening'
        cleaned_data['timestamp'] = timestamp_now.strftime("%Y-%m-%d %H:%M:%S")

        employee_id = cleaned_data.get('employee_id')
        today_date = timestamp_now.date()

        try:
            check_query = """
                SELECT timestamp FROM opening_dwm 
                WHERE employee_id = %s AND DATE(timestamp) = %s
                ORDER BY timestamp DESC LIMIT 1
            """
            cur.execute(check_query, (employee_id, today_date))
            previous_submission = cur.fetchone()

            if previous_submission:
                last_submission_time = datetime.strptime(previous_submission[0], "%Y-%m-%d %H:%M:%S")
                if (timestamp_now - last_submission_time) < timedelta(hours=8):
                    message = "⚠️ You have already submitted the form today. You can submit again after 8 hours."
                    return render_template("opening.html", message=message, alert_type="warning")

            columns = ', '.join(cleaned_data.keys())
            placeholders = ', '.join(['%s'] * len(cleaned_data))
            values = tuple(cleaned_data.values())
            insert_query = f"INSERT INTO opening_dwm ({columns}) VALUES ({placeholders})"

            cur.execute(insert_query, values)
            conn.commit()
            message = "✅ Opening data submitted successfully!"
            return render_template("opening.html", message=message, alert_type="success")

        except Exception as e:
            conn.rollback()
            message = f"❌ Error: {str(e)}"
            return render_template("opening.html", message=message, alert_type="danger")

    return render_template("opening.html")


# ------------------ Closing Form ------------------
@app.route('/dwm_report/closing', methods=['GET', 'POST'])
@login_required
def closing():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        integer_fields = [
            'grn_qty_pendency', 'stn_qty_pendency', 'putaway_cancel_qty_pendency',
            'putaway_return_qty_pendency', 'grn_sellable_qty_pendency', 'bin_movement_pendency',
            'return_pendency', 'rtv_pendency', 'channel_order_qty_b2c_pendency',
            'rts_order_qty_b2c_pendency', 'breached_qty_pendency', 'side_lined_pendency',
            'dispatch_not_marked', 'not_dispatched_orders', 'no_of_floor_associated',
            'unloading_loading_boxes', 'receipt_process_boxes', 'qty_grn_qc',
            'qty_good_putaway', 'qty_cycle_count', 'stn_direct_putaway',
            'qty_picked_b2c', 'qty_invoiced_packed_b2c', 'qty_manifest_handover_b2c',
            'qty_invoiced_packed_b2b', 'picked_qty_b2b', 'rto_received_qty',
            'rto_putaway_qty', 'qty_gp_creation_qcr', 'rto_good_processing_return',
            'bad_processing_with_claim'
        ]

        decimal_fields = [
            'unloading_loading_boxes_manpower', 'receipt_process_boxes_manpower',
            'qty_grn_qc_manpower', 'qty_good_putaway_manpower', 'qty_cycle_count_manpower',
            'stn_direct_putaway_manpower', 'qty_picked_b2c_manpower',
            'qty_invoiced_packed_b2c_manpower', 'qty_manifest_handover_b2c_manpower',
            'qty_invoiced_packed_b2b_manpower', 'picked_qty_b2b_manpower',
            'rto_received_qty_manpower', 'rto_putaway_qty_manpower',
            'qty_gp_creation_qcr_manpower', 'rto_good_processing_return_manpower',
            'bad_processing_with_claim_manpower'
        ]

        cleaned_data = {}
        for key in request.form:
            value = request.form[key].strip()
            if key in integer_fields:
                try:
                    cleaned_data[key] = int(value) if value != '' else None
                except ValueError:
                    cleaned_data[key] = None
            elif key in decimal_fields:
                try:
                    cleaned_data[key] = Decimal(value) if value != '' else None
                except (ValueError, InvalidOperation):
                    cleaned_data[key] = None
            else:
                cleaned_data[key] = value if value != '' else None

        timestamp_now = datetime.now()
        cleaned_data['source'] = 'Closing'
        cleaned_data['timestamp'] = timestamp_now.strftime("%Y-%m-%d %H:%M:%S")

        employee_id = cleaned_data.get('employee_id')
        today_date = timestamp_now.date()

        try:
            check_query = """
                SELECT timestamp FROM closing_dwm
                WHERE employee_id = %s AND DATE(timestamp) = %s
                ORDER BY timestamp DESC LIMIT 1
            """
            cur.execute(check_query, (employee_id, today_date))
            previous_submission = cur.fetchone()

            if previous_submission:
                last_time = datetime.strptime(previous_submission[0], "%Y-%m-%d %H:%M:%S")
                if (timestamp_now - last_time) < timedelta(hours=8):
                    message = "⚠️ You have already submitted the form today. You can submit again after 8 hours."
                    return render_template("closing.html", message=message, alert_type="warning")

            columns = ', '.join(cleaned_data.keys())
            placeholders = ', '.join(['%s'] * len(cleaned_data))
            values = tuple(cleaned_data.values())
            query = f"INSERT INTO closing_dwm ({columns}) VALUES ({placeholders})"

            cur.execute(query, values)
            conn.commit()
            message = "✅ Closing data submitted successfully!"
            return render_template("closing.html", message=message, alert_type="success")

        except Exception as e:
            conn.rollback()
            message = f"❌ Error: {str(e)}"
            return render_template("closing.html", message=message, alert_type="danger")

    return render_template("closing.html")





def load_and_merge_entries_from_db():
    """Fetch and merge opening_dwm and closing_dwm tables from DB."""
    try:
        conn = get_db_connection()   # ✅ ensure connection
        df_opening = pd.read_sql("SELECT * FROM opening_dwm", conn)
        df_closing = pd.read_sql("SELECT * FROM closing_dwm", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Error fetching data from DB: {e}")
        return pd.DataFrame()

    # Standardize column names
    df_opening.rename(columns=str.title, inplace=True)
    df_closing.rename(columns=str.title, inplace=True)

    # Convert date column
    df_opening["Date"] = pd.to_datetime(df_opening["Date"], errors='coerce')
    df_closing["Date"] = pd.to_datetime(df_closing["Date"], errors='coerce')

    if df_opening["Date"].isna().any() or df_closing["Date"].isna().any():
        print("⚠️ Warning: Invalid dates found!")

    # Rename columns with suffixes except merge keys
    merge_keys = ["Date", "Shift", "Location", "Customer"]
    df_opening = df_opening.rename(
        columns={col: f"{col}_opening" if col not in merge_keys else col for col in df_opening.columns}
    )
    df_closing = df_closing.rename(
        columns={col: f"{col}_closing" if col not in merge_keys else col for col in df_closing.columns}
    )

    merged_df = pd.merge(df_opening, df_closing, on=merge_keys, how="outer")
    # print(f"✅ Merged rows: {len(merged_df)}")

    # ============================
    # Role-based Location Filter
    # ============================
    try:
        from flask import session  # ✅ import inside to avoid circular import
        user_role = session.get("role")
        user_location = session.get("location")

        if user_role in ("Supervisor", "Manager") and user_location and "Location" in merged_df.columns:
            merged_df = merged_df[merged_df["Location"] == user_location]
            print(f"🔒 Role filter applied: {user_role} -> Location={user_location}")
        else:
            print(f"🔓 No role/location restriction (role={user_role})")
    except Exception as e:
        # In case function called outside Flask request context
        print(f"⚠️ Role filter skipped (no session context): {e}")

    return merged_df



@app.route('/dwm_report/dwm_data_dashboard', methods=['GET', 'POST'])
@login_required
def dwm_data_dashboard():
    df = load_and_merge_entries_from_db()

    if df.empty:
        return render_template('dwm_dashboard.html', data="<h3>No Data Available</h3>")

    # Dropdown values based on available unique combinations
    all_dates = sorted(df["Date"].dropna().dt.strftime("%Y-%m-%d").unique())
    all_locations = sorted(df["Location"].dropna().unique())
    all_customers = sorted(df["Customer"].dropna().unique())
    all_shifts = sorted(df["Shift"].dropna().unique())

    current_date = current_location = current_customer = current_shift = 'All'

    if request.method == 'POST':
        if 'clear_filters' in request.form:
            return render_template('dwm_dashboard.html',
                                   data=df.to_html(classes='table table-bordered', index=False),
                                   all_dates=all_dates,
                                   all_locations=all_locations,
                                   all_customers=all_customers,
                                   all_shifts=all_shifts,
                                   current_date='All',
                                   current_location='All',
                                   current_customer='All',
                                   current_shift='All')

        current_date = request.form.get('Date', 'All')
        current_location = request.form.get('Location', 'All')
        current_customer = request.form.get('Customer', 'All')
        current_shift = request.form.get('Shift', 'All')

        if current_date != 'All':
            df = df[df["Date"].dt.strftime("%Y-%m-%d") == current_date]
        if current_location != 'All':
            df = df[df["Location"] == current_location]
        if current_customer != 'All':
            df = df[df["Customer"] == current_customer]
        if current_shift != 'All':
            df = df[df["Shift"] == current_shift]

    return render_template('dwm_dashboard.html',
                           data=df.to_html(classes='table table-striped table-bordered', index=False),
                           all_dates=all_dates,
                           all_locations=all_locations,
                           all_customers=all_customers,
                           all_shifts=all_shifts,
                           current_date=current_date,
                           current_location=current_location,
                           current_customer=current_customer,
                           current_shift=current_shift)



@app.route('/dwm_report/dwm_dashboard_ai', methods=['GET', 'POST'])
@login_required
def dwm_dashboard_ai():
    df = load_and_merge_entries_from_db()

    master_path = os.path.join(BASE_DIR, "master_data.csv")
    try:
        master_file = pd.read_csv(master_path)
    except Exception:
        return "Error: master_data.csv not found!", 500

    if df.empty:
        return render_template("dwm_dashboard_ai.html", table_data=[], unique_dates=[], unique_shifts=[],
                               unique_locations=[], unique_customers=[])

    current_date = request.form.get("Date", "")
    current_shift = request.form.get("Shift", "")
    current_location = request.form.get("Location", "")
    current_customer = request.form.get("Customer", "")

    if request.method == 'POST':
        if 'clear_filters' in request.form:
            current_date, current_shift, current_location, current_customer = "", "", "", ""
        else:
            if current_date:
                df = df[df["Date"] == current_date]
            if current_shift:
                df = df[df["Shift"] == current_shift]
            if current_location:
                df = df[df["Location"] == current_location]
            if current_customer:
                df = df[df["Customer"] == current_customer]

    benchmark_data = [
        {"Department": "Inward", "Activity": "Unloading_Loading_Boxes", "Benchmark": 150},
        {"Department": "Inward", "Activity": "Receipt_Process_Boxes", "Benchmark": 150},
        {"Department": "Inward", "Activity": "Qty_Grn_Qc", "Benchmark": 1100},
        {"Department": "Inventory", "Activity": "Qty_Good_Putaway", "Benchmark": 1500},
        {"Department": "Inventory", "Activity": "Qty_Cycle_Count", "Benchmark": 2000},
        {"Department": "Inventory", "Activity": "Stn_Direct_Putaway", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Qty_Picked_B2C", "Benchmark": 400},
        {"Department": "Outward", "Activity": "Qty_Invoiced_Packed_B2C", "Benchmark": 450},
        {"Department": "Outward", "Activity": "Qty_Manifest_Handover_B2C", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Picked_Qty_B2B", "Benchmark": 600},
        {"Department": "Outward", "Activity": "Qty_Invoiced_Packed_B2B", "Benchmark": 800},
        {"Department": "Return", "Activity": "Rto_Received_Qty", "Benchmark": 1000},
        {"Department": "Return", "Activity": "Rto_Good_Processing_Return", "Benchmark": 200},
        {"Department": "Return", "Activity": "Rto_Putaway_Qty", "Benchmark": 1200},
        {"Department": "Return", "Activity": "Qty_Gp_Creation_Qcr", "Benchmark": 1000},
        {"Department": "Return", "Activity": "Bad_Processing_With_Claim", "Benchmark": 60},
    ]  # keep your benchmark list as-is
    benchmarks_df = pd.DataFrame(benchmark_data)
    df_cols = df.columns.tolist()

    benchmarks_df['Deployed Manpower'] = benchmarks_df['Activity'].apply(
        lambda activity: df[f"{activity}_Manpower_opening"].sum() if f"{activity}_Manpower_opening" in df_cols else 0
    )
    benchmarks_df['Target'] = benchmarks_df['Activity'].apply(
        lambda activity: df[f"{activity}_opening"].sum() if f"{activity}_opening" in df_cols else 0
    )

    pendency_fields = ['Grn_Qty_Pendency',	'Stn_Qty_Pendency',	'Putaway_Cancel_Qty_Pendency',	'Putaway_Return_Qty_Pendency',	'Grn_Sellable_Qty_Pendency',	'Bin_Movement_Pendency'
                       'Return_Pendency',	'Rtv_Pendency',	'Channel_Order_Qty_B2C_Pendency',	'Rts_Order_Qty_B2C_Pendency',	'Breached_Qty_Pendency',	'Side_Lined_Pendency',	'Dispatch_Not_Marked',	'Not_Dispatched_Orders']  # keep your list same
    # for field in pendency_fields:
    #     col = f"{field}_opening"
    #     if col in df_cols:
    #         benchmarks_df['Target'] += df[col].sum()

    benchmarks_df['Required Manpower'] = (benchmarks_df['Target'] / benchmarks_df['Benchmark']).replace(
        [np.inf, -np.inf], 0).fillna(0).round(2)
    benchmarks_df['Extra Manpower'] = (benchmarks_df['Deployed Manpower'] - benchmarks_df['Required Manpower']).round(2)

    benchmarks_df['Execution'] = benchmarks_df['Activity'].apply(
        lambda activity: df[f"{activity}_closing"].sum() if f"{activity}_closing" in df_cols else 0
    )
    # benchmarks_df['Pendency'] = (benchmarks_df['Target'] - benchmarks_df['Execution']).round(2)
    benchmarks_df['Capacity'] = (benchmarks_df['Deployed Manpower'] * benchmarks_df['Benchmark']).round(2)

    benchmarks_df['Capacity Vs Execution'] = benchmarks_df.apply(
        lambda row: f"{(row['Execution'] / row['Capacity'] * 100):.2f}%" if row['Capacity'] > 0 else "0.00%", axis=1
    )
    benchmarks_df['Target Vs Execution'] = benchmarks_df.apply(
        lambda row: f"{(row['Execution'] / row['Target'] * 100):.2f}%" if row['Target'] > 0 else "0.00%", axis=1
    )

    # 🔁 Merge with master
    table_data_df = benchmarks_df.merge(master_file, on="Activity", how="left")
    table_data_df.rename(columns={'Department_x': 'Department', 'Benchmark_x': 'Benchmark'}, inplace=True)

    # ✅ Define numeric columns for totaling
    numeric_columns = [
        'Head Count', 'Planned Load', 'Deployed Manpower',
        'Target', 'Required Manpower', 'Extra Manpower',
        'Execution', 'Capacity'
    ]

    # ✅ Calculate totals
    totals = {}
    for col in numeric_columns:
        totals[col] = table_data_df[col].sum().round(2)

    # Fill non-numeric display columns
    totals['Department'] = 'Total'
    totals['Activity'] = ''
    totals['Benchmark'] = ''

    # For percentage fields, calculate weighted average or show blank
    totals['Capacity Vs Execution'] = ''
    totals['Target Vs Execution'] = ''

    # Total Opening & Closing Pendency
    total_opening_pendency = 0
    total_closing_pendency = 0

    for field in pendency_fields:
        opening_col = f"{field}_opening"
        closing_col = f"{field}_closing"

        if opening_col in df_cols:
            total_opening_pendency += df[opening_col].sum()

        if closing_col in df_cols:
            total_closing_pendency += df[closing_col].sum()

    # ✅ Total Target from benchmarks_df
    total_target = benchmarks_df['Target'].sum()

    # ✅ Opening + Target
    total_opening_plus_target = total_opening_pendency + total_target

    column_order = [
        'Department', 'Activity', 'Benchmark', 'Head Count', 'Planned Load',
        'Deployed Manpower', 'Target', 'Required Manpower', 'Extra Manpower',
        'Execution', 'Capacity', 'Capacity Vs Execution', 'Target Vs Execution'
    ]
    table_data_df = table_data_df[column_order]
    final_table_data = table_data_df.to_dict(orient='records')
    session['table_data_list'] = final_table_data

    # ✅ Pendency Modal Logic
    pendency_data = []
    if current_date:
        df['Date'] = pd.to_datetime(df['Date'])
        current_date_dt = pd.to_datetime(current_date)
        previous_date = (current_date_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        df_today = df[df['Date'] == current_date_dt]
        df_yesterday = df[df['Date'] == pd.to_datetime(previous_date)]

        common_cols = [col.replace("_opening", "") for col in df.columns if col.endswith("_opening")]
        for col in common_cols:
            # Only include selected pendency fields
            if col not in pendency_fields:
                continue

            today_col = f"{col}_opening"
            yesterday_col = f"{col}_closing"

            # ✅ Convert to numeric safely
            today_val = pd.to_numeric(df_today[today_col], errors='coerce').sum() if today_col in df_today else 0
            yesterday_val = pd.to_numeric(df_yesterday[yesterday_col],
                                          errors='coerce').sum() if yesterday_col in df_yesterday else 0

            pendency_data.append({
                "Metric": col.replace("_", " ").title(),
                "Yesterday_Closing": int(yesterday_val),
                "Today_Opening": int(today_val),
                "Difference": int(today_val - yesterday_val)
            })

    # ✅ Predicted Manpower Logic
    predicted_data = []
    for _, row in benchmarks_df.iterrows():
        predicted_data.append({
            "Activity": row["Activity"],
            "Target": row["Target"],
            "Benchmark": row["Benchmark"],
            "Predicted Manpower": row["Required Manpower"]
        })

    return render_template("dwm_dashboard_ai.html",
                           table_data=final_table_data,
                           totals=totals,
                           unique_dates=df["Date"].dt.strftime('%Y-%m-%d').unique().tolist(),
                           unique_shifts=df["Shift"].unique().tolist(),
                           unique_locations=df["Location"].unique().tolist(),
                           unique_customers=df["Customer"].unique().tolist(),
                           selected_date=current_date,
                           selected_shift=current_shift,
                           selected_location=current_location,
                           selected_customer=current_customer,
                           raw_data=df.to_dict(orient="records"),
                           pendency_data=pendency_data,
                           predicted_data=predicted_data,
                           total_opening_pendency=int(total_opening_pendency),
                           total_closing_pendency=int(total_closing_pendency),
                           total_target=int(total_target),
                           total_opening_plus_target=int(total_opening_plus_target)
                           )


@app.route("/download")
@login_required
def download_report():
    df = pd.DataFrame(session.get('table_data_list', []))
    if df.empty:
        return "No data to download"
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="DWM_Report.csv")


@app.route('/dwm_report/upload_master', methods=['GET', 'POST'])
@login_required
def upload_master():
    if request.method == 'POST':
        file = request.files.get('master_file')
        if file:
            df = pd.read_csv(file)

            # Clear existing master table
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM master_data;")

            # Insert rows
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO master_data (Activity, Department, Head_Count, Planned_Load)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (Activity) DO UPDATE
                    SET Department = EXCLUDED.Department,
                        Head_Count = EXCLUDED.Head_Count,
                        Planned_Load = EXCLUDED.Planned_Load;
                """, (row['Activity'], row['Department'], row.get('Head_Count', 0), row.get('Planned_Load', 0)))

            conn.commit()
            cur.close()
            conn.close()

            flash("✅ Master file uploaded to DB successfully!", "success")
            return redirect(url_for('performance_metrics'))
        else:
            flash("❌ No file selected!", "danger")
    return render_template("upload_master.html")



def load_master_from_db():
    try:
        conn = get_db_connection()
        query = "SELECT activity, department, head_count, planned_load FROM master_data"
        master_df = pd.read_sql(query, conn)
        conn.close()
        return master_df
    except Exception as e:
        print(f"Error loading master file from DB: {e}")
        return pd.DataFrame()  # fallback


@app.route('/performance-metrics', methods=['GET', 'POST'])
@login_required
def performance_metrics():
    import numpy as np
    import pandas as pd
    import os

    df = load_and_merge_entries_from_db()

    master_path = os.path.join(BASE_DIR, "master_data.csv")
    try:
        master_file = pd.read_csv(master_path)
    except Exception:
        return "Error: master_data.csv not found!", 500

    if df.empty:
        return render_template("performance_metrics.html",
                               table_data=[],
                               unique_dates=[],
                               unique_shifts=[],
                               unique_locations=[],
                               unique_customers=[],
                               unique_departments=[],
                               daily_totals=[],
                               monthly_totals=[])

    # ---- Filters ----
    current_date = request.form.get("Date", "")
    current_shift = request.form.get("Shift", "")
    current_location = request.form.get("Location", "")
    current_customer = request.form.get("Customer", "")
    current_department = request.form.get("Department", "")
    smart_date = request.form.get("SmartDate", "")

    if request.method == 'POST':
        if 'clear_filters' in request.form:
            current_date = current_shift = current_location = current_customer = current_department = smart_date = ""
        else:
            if current_date:
                df = df[df["Date"] == current_date]
            if current_shift:
                df = df[df["Shift"] == current_shift]
            if current_location:
                df = df[df["Location"] == current_location]
            if current_customer:
                df = df[df["Customer"] == current_customer]
            if current_department:
                dept_activities = master_file[master_file["Department"] == current_department]["Activity"].tolist()
                df = df[[col for col in df.columns if any(act in col for act in dept_activities)] +
                        ["Date", "Shift", "Location", "Customer"]]

            if smart_date:
                today = pd.to_datetime("today").normalize()
                if smart_date == "today":
                    df = df[df["Date"] == today]
                elif smart_date == "yesterday":
                    df = df[df["Date"] == (today - pd.Timedelta(days=1))]
                elif smart_date == "week":
                    start_week = today - pd.Timedelta(days=today.weekday())
                    df = df[df["Date"].between(start_week, today)]
                elif smart_date == "month":
                    start_month = today.replace(day=1)
                    df = df[df["Date"].between(start_month, today)]

    # ---- Benchmark Data ----
    benchmark_data = [
        {"Department": "Inward", "Activity": "Unloading_Loading_Boxes", "Benchmark": 150},
        {"Department": "Inward", "Activity": "Receipt_Process_Boxes", "Benchmark": 150},
        {"Department": "Inward", "Activity": "Qty_Grn_Qc", "Benchmark": 1100},
        {"Department": "Inventory", "Activity": "Qty_Good_Putaway", "Benchmark": 1500},
        {"Department": "Inventory", "Activity": "Qty_Cycle_Count", "Benchmark": 2000},
        {"Department": "Inventory", "Activity": "Stn_Direct_Putaway", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Qty_Picked_B2C", "Benchmark": 400},
        {"Department": "Outward", "Activity": "Qty_Invoiced_Packed_B2C", "Benchmark": 450},
        {"Department": "Outward", "Activity": "Qty_Manifest_Handover_B2C", "Benchmark": 1500},
        {"Department": "Outward", "Activity": "Picked_Qty_B2B", "Benchmark": 600},
        {"Department": "Outward", "Activity": "Qty_Invoiced_Packed_B2B", "Benchmark": 800},
        {"Department": "Return", "Activity": "Rto_Received_Qty", "Benchmark": 1000},
        {"Department": "Return", "Activity": "Rto_Good_Processing_Return", "Benchmark": 200},
        {"Department": "Return", "Activity": "Rto_Putaway_Qty", "Benchmark": 1200},
        {"Department": "Return", "Activity": "Qty_Gp_Creation_Qcr", "Benchmark": 1000},
        {"Department": "Return", "Activity": "Bad_Processing_With_Claim", "Benchmark": 60},
    ]
    benchmarks_df = pd.DataFrame(benchmark_data)
    df_cols = df.columns.tolist()

    # ---- Calculations ----
    benchmarks_df['Actual Manpower'] = benchmarks_df['Activity'].apply(
        lambda activity: df[f"{activity}_Manpower_opening"].sum() if f"{activity}_Manpower_opening" in df_cols else 0
    )
    benchmarks_df['Target'] = benchmarks_df['Activity'].apply(
        lambda activity: df[f"{activity}_opening"].sum() if f"{activity}_opening" in df_cols else 0
    )
    benchmarks_df['Required Manpower'] = (benchmarks_df['Target'] / benchmarks_df['Benchmark']).replace(
        [np.inf, -np.inf], 0).fillna(0).round(2)
    benchmarks_df['Execution'] = benchmarks_df['Activity'].apply(
        lambda activity: df[f"{activity}_closing"].sum() if f"{activity}_closing" in df_cols else 0
    )
    benchmarks_df['Capacity'] = (benchmarks_df['Actual Manpower'] * benchmarks_df['Benchmark']).round(2)

    benchmarks_df['Capacity Utilization %'] = benchmarks_df.apply(
        lambda row: f"{(row['Execution'] / row['Capacity'] * 100):.2f}%" if row['Capacity'] > 0 else "0.00%", axis=1
    )
    benchmarks_df['Manpower Utilization %'] = benchmarks_df.apply(
        lambda row: f"{(row['Execution'] / (row['Required Manpower'] * row['Benchmark']) * 100):.2f}%"
        if row['Required Manpower'] > 0 else "0.00%", axis=1
    )

    # ---- Merge with Master ----
    table_data_df = benchmarks_df.merge(master_file, on="Activity", how="left")
    table_data_df.rename(columns={'Department_x': 'Department', 'Benchmark_x': 'Benchmark'}, inplace=True)

    # ---- Column Ordering ----
    column_order = [
        'Department', 'Activity', 'Benchmark',
        'Head Count', 'Planned Load', 'Actual Manpower',
        'Target', 'Required Manpower', 'Execution', 'Capacity',
        'Capacity Utilization %', 'Manpower Utilization %'
    ]
    table_data_df = table_data_df[column_order]

    # ---- Dict for Template ----
    final_table_data = table_data_df.to_dict(orient='records')

    # ---- Daily + Monthly totals ----
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    daily_totals = []
    for day, group in df.groupby(df['Date'].dt.strftime("%Y-%m-%d")):
        daily_table = benchmarks_df.copy()
        daily_cols = group.columns.tolist()

        daily_table['Actual Manpower'] = daily_table['Activity'].apply(
            lambda activity: group[f"{activity}_Manpower_opening"].sum() if f"{activity}_Manpower_opening" in daily_cols else 0
        )
        daily_table['Target'] = daily_table['Activity'].apply(
            lambda activity: group[f"{activity}_opening"].sum() if f"{activity}_opening" in daily_cols else 0
        )
        daily_table['Required Manpower'] = (daily_table['Target'] / daily_table['Benchmark']).replace(
            [np.inf, -np.inf], 0).fillna(0).round(2)
        daily_table['Execution'] = daily_table['Activity'].apply(
            lambda activity: group[f"{activity}_closing"].sum() if f"{activity}_closing" in daily_cols else 0
        )
        daily_table['Capacity'] = (daily_table['Actual Manpower'] * daily_table['Benchmark']).round(2)

        daily_totals.append({
            "Day": day,
            "Head Count": group["Head Count"].sum() if "Head Count" in group.columns else 0,
            "Planned Load": group["Planned Load"].sum() if "Planned Load" in group.columns else 0,
            "Actual Manpower": daily_table['Actual Manpower'].sum(),
            "Target": daily_table['Target'].sum(),
            "Required Manpower": daily_table['Required Manpower'].sum(),
            "Execution": daily_table['Execution'].sum(),
            "Capacity": daily_table['Capacity'].sum(),
            "Capacity Utilization %": f"{(daily_table['Execution'].sum() / daily_table['Capacity'].sum() * 100):.2f}%"
            if daily_table['Capacity'].sum() > 0 else "0.00%",
            "Manpower Utilization %": (
                f"{(daily_table['Execution'].sum() / (daily_table['Required Manpower'].sum() * daily_table['Benchmark'].mean()) * 100):.2f}%"
                if daily_table['Required Manpower'].sum() > 0 and daily_table['Benchmark'].mean() > 0 else "0.00%"
            ),
        })

    monthly_totals = []
    for month, group in df.groupby(df['Date'].dt.to_period("M").astype(str)):
        monthly_table = benchmarks_df.copy()
        monthly_cols = group.columns.tolist()

        monthly_table['Actual Manpower'] = monthly_table['Activity'].apply(
            lambda activity: group[f"{activity}_Manpower_opening"].sum() if f"{activity}_Manpower_opening" in monthly_cols else 0
        )
        monthly_table['Target'] = monthly_table['Activity'].apply(
            lambda activity: group[f"{activity}_opening"].sum() if f"{activity}_opening" in monthly_cols else 0
        )
        monthly_table['Required Manpower'] = (monthly_table['Target'] / monthly_table['Benchmark']).replace(
            [np.inf, -np.inf], 0).fillna(0).round(2)
        monthly_table['Execution'] = monthly_table['Activity'].apply(
            lambda activity: group[f"{activity}_closing"].sum() if f"{activity}_closing" in monthly_cols else 0
        )
        monthly_table['Capacity'] = (monthly_table['Actual Manpower'] * monthly_table['Benchmark']).round(2)

        monthly_totals.append({
            "Month": month,
            "Head Count": group["Head Count"].sum() if "Head Count" in group.columns else 0,
            "Planned Load": group["Planned Load"].sum() if "Planned Load" in group.columns else 0,
            "Actual Manpower": monthly_table['Actual Manpower'].sum(),
            "Target": monthly_table['Target'].sum(),
            "Required Manpower": monthly_table['Required Manpower'].sum(),
            "Execution": monthly_table['Execution'].sum(),
            "Capacity": monthly_table['Capacity'].sum(),
            "Capacity Utilization %": f"{(monthly_table['Execution'].sum() / monthly_table['Capacity'].sum() * 100):.2f}%"
            if monthly_table['Capacity'].sum() > 0 else "0.00%",
            "Manpower Utilization %": (
                f"{(monthly_table['Execution'].sum() / (monthly_table['Required Manpower'].sum() * monthly_table['Benchmark'].mean()) * 100):.2f}%"
                if monthly_table['Required Manpower'].sum() > 0 and monthly_table['Benchmark'].mean() > 0 else "0.00%"
            ),
        })

    # ---- Convert numpy types to Python native ----
    def convert_numpy_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    final_table_data = convert_numpy_to_python(final_table_data)
    daily_totals = convert_numpy_to_python(daily_totals)
    monthly_totals = convert_numpy_to_python(monthly_totals)

    # ---- Render Template ----
    return render_template("performance_metrics.html",
                           table_data=final_table_data,
                           unique_dates=df["Date"].dt.strftime('%Y-%m-%d').dropna().unique().tolist(),
                           unique_shifts=df["Shift"].dropna().unique().tolist(),
                           unique_locations=df["Location"].dropna().unique().tolist(),
                           unique_customers=df["Customer"].dropna().unique().tolist(),
                           unique_departments=master_file["Department"].dropna().unique().tolist(),
                           selected_date=current_date,
                           selected_shift=current_shift,
                           selected_location=current_location,
                           selected_customer=current_customer,
                           selected_department=current_department,
                           smart_date=smart_date,
                           daily_totals=daily_totals,
                           monthly_totals=monthly_totals
                           )



@app.route('/sla_report/unicom')
def unicom():
    return render_template('unicom.html')

@app.route('/sla_report/eshopbox')
def eshopbox():
    return render_template('eshopbox.html')

@app.route('/sla_report/pendency')
def pendency_sla():
    return render_template('pendency_sla.html')

@app.route('/sla_report/unicom_report')
def unicom_report():
    return render_template('unicom_report.html')

# SLA Report Route
@app.route('/sla_report')
def sla_report():
    return render_template('sla_report.html')

# KPI Report Route
@app.route('/kpi_report')
def kpi_report():
    return render_template('kpi_report.html')


# HRMS Report Route
@app.route('/hrms_report')
def hrms_report():
    return render_template('hrms_report.html')





# Download Dashboard Route


if __name__ == '__main__':
    app.run(debug=True)
