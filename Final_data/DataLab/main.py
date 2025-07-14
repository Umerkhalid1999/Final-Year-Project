# app.py - Main Flask application with Firebase Authentication
import os
import json
import logging
from datetime import datetime
from functools import wraps

import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth
from flask_cors import CORS  # Add this import

# todo: Data preprocessing + routes implementation
from pycaret.classification import *
from pycaret.regression import *
from pycaret.clustering import *
import tempfile
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, make_response
)
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datalab.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Application Configuration
class Config:
    SECRET_KEY = os.urandom(32)
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'xlsx', 'jpg', 'png'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    FIREBASE_CONFIG_PATH = os.environ.get(
        'FIREBASE_CONFIG_PATH',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'data-storing123-firebase-adminsdk-fbsvc-2a77c2f29a.json')
    )


# Initialize Firebase Admin SDK
def initialize_firebase():
    try:
        cred = credentials.Certificate(Config.FIREBASE_CONFIG_PATH)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.error(f"Firebase initialization error: {e}")
        raise


# Create Flask Application
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for development

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Firebase
initialize_firebase()

# In-memory storage for datasets
datasets = {}
next_dataset_id = 1


# Helper Functions
def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Firebase Authentication Helper
def verify_firebase_token(id_token):
    """
    Verify Firebase ID token and return user information

    Args:
        id_token (str): Firebase ID token

    Returns:
        dict: User information or None if token is invalid
    """
    try:
        # Verify and decode the ID token
        decoded_token = auth.verify_id_token(id_token)

        # Get user information
        user = auth.get_user(decoded_token['uid'])

        return {
            'uid': user.uid,
            'email': user.email,
            'display_name': user.display_name
        }
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


# Authentication Middleware
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the token from cookies
        token = request.cookies.get('token')
        logger.info(f"login_required - Cookie token present: {token is not None}")

        # If no token, redirect to login
        if not token:
            logger.warning("No token in cookies, redirecting to login")
            return redirect(url_for('login'))

        try:
            # Verify the token
            user_info = verify_firebase_token(token)

            if not user_info:
                logger.warning("Token verification failed")
                # Clear invalid token
                response = make_response(redirect(url_for('login')))
                response.set_cookie('token', '', expires=0)
                return response

            logger.info(f"User authenticated: {user_info['email']}")

            # Set user info in session if not already there
            if 'user_id' not in session or session['user_id'] != user_info['uid']:
                logger.info(f"Setting session for user: {user_info['uid']}")
                session['user_id'] = user_info['uid']
                session['email'] = user_info['email']
                session['username'] = user_info['display_name'] or user_info['email'].split('@')[0]

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            # Clear session and redirect to login
            session.clear()
            return redirect(url_for('login'))

    return decorated_function


# Routes
@app.route('/')
def index():
    """Main landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login')
def login():
    """Login page"""
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/signup')
def signup():
    """Signup page"""
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    return render_template('signup.html')


@app.route('/forgot-password')
def forgot_password():
    """Forgot password page"""
    return render_template('forgot_password.html')


@app.route('/reset-password')
def reset_password():
    """Reset password page"""
    return render_template('reset_password.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    user_id = session['user_id']
    # Debug logging
    logger.info(f"Dashboard accessed by user: {user_id}")
    logger.info(f"Available datasets: {datasets.get(user_id, [])}")

    # Get user's datasets from memory
    user_datasets = datasets.get(user_id, [])

    return render_template(
        'dashboard.html',
        username=session.get('username', ''),
        datasets=user_datasets,
        datasets_count=len(user_datasets)
    )


@app.route('/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    global next_dataset_id

    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})

    if file and allowed_file(file.filename):
        try:
            # Save the file
            filename = secure_filename(file.filename)

            # Add timestamp to filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(file_path)

            # Get file type
            file_type = filename.rsplit('.', 1)[1].lower()

            # Analyze file
            analysis = analyze_file(file_path, file_type)

            # Create new dataset entry
            user_id = session['user_id']
            dataset_id = next_dataset_id
            next_dataset_id += 1

            # Make sure the user has a dataset list
            if user_id not in datasets:
                datasets[user_id] = []

            # Deactivate all other datasets
            for ds in datasets[user_id]:
                ds['active'] = False

            # Create dataset object
            new_dataset = {
                "id": dataset_id,
                "name": filename,
                "file_type": file_type,
                "file_path": file_path,
                "rows": analysis.get('rows', 0),
                "columns": analysis.get('columns', 0),
                "quality_score": analysis.get('quality_score', 0),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "active": True
            }

            # Add to user's datasets
            datasets[user_id].append(new_dataset)

            # Return success with dataset info
            return jsonify({
                "success": True,
                "message": "File uploaded successfully",
                "dataset_id": new_dataset["id"],
                "dataset": {
                    "id": new_dataset["id"],
                    "name": new_dataset["name"],
                    "file_type": new_dataset["file_type"],
                    "rows": new_dataset["rows"],
                    "columns": new_dataset["columns"]
                }
            })

        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return jsonify({"success": False, "message": f"Error processing file: {str(e)}"})

    return jsonify({"success": False, "message": "File type not allowed"})


@app.route('/api/datasets')
@login_required
def get_datasets():
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    return jsonify(user_datasets)


@app.route('/data_preview/<int:dataset_id>')
@login_required
def data_preview(dataset_id):
    """Preview dataset content"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        flash('Dataset not found')
        return redirect(url_for('dashboard'))

    try:
        # Read file based on type
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type == 'csv':
            df = pd.read_csv(file_path)
            headers = df.columns.tolist()
            data = df.head(10).values.tolist()
        elif file_type == 'json':
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            # Convert JSON to a dataframe for display
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            else:
                df = pd.DataFrame([json_data])
            headers = df.columns.tolist()
            data = df.head(10).values.tolist()
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
            headers = df.columns.tolist()
            data = df.head(10).values.tolist()
        else:
            # For other file types, show message
            headers = ['Content']
            with open(file_path, 'r') as f:
                content = f.read(1000)  # Read first 1000 chars
            data = [[content]]

        return render_template(
            'data_preview.html',
            dataset=dataset,
            headers=headers,
            data=data
        )

    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        flash(f'Error previewing data: {str(e)}')
        return redirect(url_for('dashboard'))


@app.route('/data_quality/<int:dataset_id>')
@login_required
def data_quality(dataset_id):
    """Data quality assessment"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        flash('Dataset not found')
        return redirect(url_for('dashboard'))

    try:
        # Read file based on type
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        quality_metrics = {}

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Missing values
            missing_values = df.isnull().sum().sum()
            missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100

            # Duplicated rows
            duplicate_rows = df.duplicated().sum()

            # Data types
            dtypes = df.dtypes.value_counts().to_dict()
            dtype_counts = {str(k): int(v) for k, v in dtypes.items()}

            # Outliers in numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_info = {}

            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_step = 1.5 * IQR
                    outliers = ((df[col] < (Q1 - outlier_step)) | (df[col] > (Q3 + outlier_step))).sum()
                    outliers_info[col] = int(outliers)

            quality_metrics = {
                'missing_values': int(missing_values),
                'missing_percentage': round(missing_percentage, 2),
                'duplicate_rows': int(duplicate_rows),
                'data_types': dtype_counts,
                'outliers': outliers_info,
                'quality_score': dataset['quality_score']
            }

        return render_template(
            'data_quality.html',
            dataset=dataset,
            quality_metrics=quality_metrics
        )

    except Exception as e:
        logger.error(f"Error analyzing data quality: {e}")
        flash(f'Error analyzing data quality: {str(e)}')
        return redirect(url_for('dashboard'))


@app.route('/delete_dataset/<int:dataset_id>', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    """Delete a dataset"""
    user_id = session['user_id']

    if user_id not in datasets:
        return jsonify({"success": False, "message": "No datasets found"})

    # Find dataset by id
    dataset_index = next((i for i, ds in enumerate(datasets[user_id]) if ds['id'] == dataset_id), None)

    if dataset_index is None:
        return jsonify({"success": False, "message": "Dataset not found"})

    try:
        # Get file path to delete
        file_path = datasets[user_id][dataset_index]['file_path']

        # Remove file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove dataset from list
        del datasets[user_id][dataset_index]

        return jsonify({"success": True, "message": "Dataset deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        return jsonify({"success": False, "message": f"Error deleting dataset: {str(e)}"})


# Enhanced version of the summary_statistics route to support user preferences
# Add this to your main.py file to replace the existing route

@app.route('/summary_statistics/<int:dataset_id>')
@login_required
def summary_statistics(dataset_id):
    """Show summary statistics for dataset with user preferences"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Get user preferences from query parameters (if any)
    selected_stats = request.args.getlist('stats')

    # Default statistics if none selected
    if not selected_stats:
        selected_stats = ['mean', 'median', 'std', 'min', 'max', 'range', 'quartiles']

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        flash('Dataset not found')
        return redirect(url_for('dashboard'))

    try:
        # Read file based on type
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Get numeric columns
            numeric_df = df.select_dtypes(include=[np.number])

            # Basic statistics with describe()
            basic_stats = numeric_df.describe().to_dict()

            # Advanced statistics based on user preferences
            formatted_summary = {}
            for col, stats in basic_stats.items():
                formatted_summary[col] = {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}

                # Add additional statistics based on user preferences
                if 'skew' in selected_stats:
                    formatted_summary[col]['skew'] = round(numeric_df[col].skew(), 3)

                if 'kurtosis' in selected_stats:
                    formatted_summary[col]['kurtosis'] = round(numeric_df[col].kurtosis(), 3)

                if 'sem' in selected_stats:
                    formatted_summary[col]['sem'] = round(numeric_df[col].sem(), 3)

                if 'cv' in selected_stats and formatted_summary[col]['mean'] != 0:
                    # Coefficient of variation (CV) = std / mean * 100
                    formatted_summary[col]['cv'] = round(
                        (formatted_summary[col]['std'] / formatted_summary[col]['mean']) * 100, 2)

                if 'sum' in selected_stats:
                    formatted_summary[col]['sum'] = round(numeric_df[col].sum(), 2)

                if 'mode' in selected_stats:
                    # Get the most common value as mode
                    mode_value = numeric_df[col].mode().iloc[0] if not numeric_df[col].mode().empty else None
                    formatted_summary[col]['mode'] = round(mode_value, 2) if isinstance(mode_value,
                                                                                        float) else mode_value

                if 'var' in selected_stats:
                    formatted_summary[col]['var'] = round(numeric_df[col].var(), 2)

                if 'iqr' in selected_stats:
                    # Interquartile range (IQR) = Q3 - Q1
                    formatted_summary[col]['iqr'] = round(formatted_summary[col]['75%'] - formatted_summary[col]['25%'],
                                                          2)

                # Add custom percentiles if requested
                if 'percentiles' in selected_stats:
                    percentiles = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
                    for p in percentiles:
                        p_str = f"{int(p * 100)}%"
                        if p_str not in formatted_summary[col]:
                            formatted_summary[col][p_str] = round(numeric_df[col].quantile(p), 2)

            return render_template(
                'summary_statistics.html',
                dataset=dataset,
                summary=formatted_summary,
                selected_stats=selected_stats
            )
        else:
            flash('Summary statistics are only available for CSV and Excel files')
            return redirect(url_for('dashboard'))

    except Exception as e:
        logger.error(f"Error calculating summary statistics: {e}")
        flash(f'Error calculating summary statistics: {str(e)}')
        return redirect(url_for('dashboard'))


# Add these imports if they're not already at the top of your file
import time
from datetime import datetime
from openai_api import OpenAIAPI

# Helper function (to replace the require_api_key decorator)
def require_openai_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return jsonify({"success": False, "error": "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."}), 500
        return f(*args, **kwargs)
    return decorated_function

# Add these AI assistant routes to your main.py file
@app.route('/ai_assistant')
@login_required
def ai_assistant_page():
    """AI Assistant dedicated page"""
    user_id = session['user_id']
    return render_template(
        'ai_assistant.html',
        username=session.get('username', '')
    )


@app.route('/api/ai_assistant/chat', methods=['POST'])
@login_required
def ai_assistant_chat():
    """Process a chat request to the AI assistant using OpenAI"""
    try:
        user_id = session['user_id']
        data = request.json
        user_message = data.get('message', '')

        # Initialize history for this user if not exists
        if 'ai_chat_history' not in session:
            session['ai_chat_history'] = []

        # Initialize OpenAI API
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return jsonify({
                "message": "OpenAI API key not configured. Please set your OPENAI_API_KEY environment variable.",
                "success": False,
                "error": "API key not configured"
            }), 500

        openai_client = OpenAIAPI(api_key)

        # Simple fallback mechanism in case API fails
        try:
            # Call API and get response
            response = openai_client.chat_response(user_message)
            
            if 'error' in response:
                # Handle API errors gracefully
                ai_content = f"I'm experiencing some technical difficulties: {response['error']}. Please try again later."
            else:
                ai_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')

            if not ai_content:
                # Fallback response if empty
                ai_content = "I'm having trouble generating a response. Please try rephrasing your question."
                
        except Exception as api_error:
            logger.error(f"OpenAI API error: {str(api_error)}")
            # Fallback if API fails entirely
            ai_content = "I'm currently experiencing technical difficulties. Here are some general data tips: 1) Always check for missing values, 2) Consider normalizing numeric features, 3) Handle outliers appropriately."

        # Store conversation in session history
        session['ai_chat_history'].append({
            'user': user_message,
            'assistant': ai_content,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({
            "message": ai_content,
            "success": True
        })

    except Exception as e:
        logger.error(f"AI Assistant error: {str(e)}")
        return jsonify({
            "message": "Sorry, I encountered an error processing your request. Please try again.",
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/ai_assistant/history', methods=['GET'])
@login_required
def ai_assistant_history():
    """Get AI chat history for the current user"""
    history = session.get('ai_chat_history', [])
    return jsonify({"success": True, "history": history})

@app.route('/api/ai_assistant/clear_history', methods=['POST'])
@login_required
def ai_assistant_clear_history():
    """Clear AI chat history for the current user"""
    session['ai_chat_history'] = []
    return jsonify({"success": True, "message": "Chat history cleared"})

# API endpoints for Firebase authentication
@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a session from Firebase ID token"""
    logger.info("Session creation request received")

    # Log request headers and body for debugging
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request content type: {request.content_type}")

    try:
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return jsonify({"success": False, "message": "Invalid JSON data"}), 400

    if not data or 'idToken' not in data:
        logger.warning("No ID token provided in request")
        return jsonify({"success": False, "message": "No ID token provided"}), 400

    id_token = data['idToken']

    try:
        # Verify the ID token
        user_info = verify_firebase_token(id_token)

        if not user_info:
            logger.warning("Invalid token provided")
            return jsonify({"success": False, "message": "Invalid token"}), 401

        # Set user info in session
        session['user_id'] = user_info['uid']
        session['email'] = user_info['email']
        session['username'] = user_info['display_name'] or user_info['email'].split('@')[0]

        logger.info(f"Session created for user: {user_info['uid']}")

        # Create response with cookie
        response = jsonify({"success": True, "message": "Session created", "user": user_info})

        # Set cookie with the token - use secure=False in development
        max_age = 3600  # 1 hour
        response.set_cookie('token', id_token, max_age=max_age, httponly=True, secure=False, samesite='Strict')

        return response

    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({"success": False, "message": f"Authentication error: {str(e)}"}), 401


@app.route('/logout')
def logout():
    """User logout route"""
    user_email = session.get('email')
    session.clear()

    if user_email:
        logger.info(f"User {user_email} logged out")

    # Create response that clears the token cookie
    response = make_response(redirect(url_for('login')))
    response.set_cookie('token', '', expires=0)

    return response


# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler"""
    logger.error(f"Internal Server Error: {e}")
    return render_template('500.html'), 500


# Utility function for file analysis
def analyze_file(file_path, file_type):
    """
    Analyze uploaded file and return basic metrics with detailed quality scoring

    Args:
        file_path (str): Path to the uploaded file
        file_type (str): Type of the file

    Returns:
        dict: File analysis metrics including detailed quality score breakdown
    """
    data = {}

    try:
        if file_type == 'csv':
            # Read CSV file
            df = pd.read_csv(file_path)

            # Basic metrics
            data['rows'] = len(df)
            data['columns'] = len(df.columns)

            # Calculate quality metrics

            # 1. Missing values (100% weight of their percentage)
            missing_values = df.isnull().sum().sum()
            total_values = data['rows'] * data['columns']
            missing_percentage = (missing_values / total_values) * 100 if total_values > 0 else 0
            missing_penalty = missing_percentage  # Full weight (100%)
            data['missing_values'] = int(missing_values)
            data['missing_percentage'] = round(missing_percentage, 2)

            # 2. Duplicate rows (50% weight of their percentage)
            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / data['rows']) * 100 if data['rows'] > 0 else 0
            duplicate_penalty = duplicate_percentage * 0.5  # 50% weight
            data['duplicate_rows'] = int(duplicate_rows)
            data['duplicate_percentage'] = round(duplicate_percentage, 2)

            # 3. Outliers in numerical columns (25% weight of their percentage)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_count = 0
            outliers_info = {}

            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_step = 1.5 * IQR
                    outliers_in_col = ((df[col] < (Q1 - outlier_step)) | (df[col] > (Q3 + outlier_step))).sum()
                    outliers_count += outliers_in_col
                    outliers_info[col] = int(outliers_in_col)

            # Safety check to avoid division by zero
            if len(numeric_cols) > 0 and data['rows'] > 0:
                outlier_percentage = (outliers_count / (len(numeric_cols) * data['rows'])) * 100
            else:
                outlier_percentage = 0

            outlier_penalty = outlier_percentage * 0.25  # 25% weight
            data['outliers'] = outliers_info
            data['outliers_count'] = int(outliers_count)
            data['outlier_percentage'] = round(outlier_percentage, 2)

            # 4. Data type issues (future enhancement)
            # This would check for columns that might have inappropriate data types

            # Calculate final quality score
            total_penalty = missing_penalty + duplicate_penalty + outlier_penalty
            data['quality_score'] = round(max(0, 100 - total_penalty), 1)

            # Store quality score components for transparency
            data['quality_components'] = {
                'missing_values': {
                    'count': int(missing_values),
                    'percentage': round(missing_percentage, 2),
                    'penalty': round(missing_penalty, 2),
                    'weight': 1.0  # 100% weight
                },
                'duplicate_rows': {
                    'count': int(duplicate_rows),
                    'percentage': round(duplicate_percentage, 2),
                    'penalty': round(duplicate_penalty, 2),
                    'weight': 0.5  # 50% weight
                },
                'outliers': {
                    'count': int(outliers_count),
                    'percentage': round(outlier_percentage, 2),
                    'penalty': round(outlier_penalty, 2),
                    'weight': 0.25  # 25% weight
                }
            }

        elif file_type == 'json':
            # Read JSON file
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            if isinstance(json_data, list):
                data['rows'] = len(json_data)
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    data['columns'] = len(json_data[0].keys())
                else:
                    data['columns'] = 1
            else:
                data['rows'] = 1
                data['columns'] = len(json_data.keys()) if isinstance(json_data, dict) else 1

            # Simple quality score for JSON
            data['quality_score'] = 90  # Default value for JSON
            data['quality_components'] = {
                'note': 'Detailed quality analysis not available for JSON files'
            }

        elif file_type in ['xlsx', 'xls']:
            # Basic Excel file analysis
            df = pd.read_excel(file_path)
            data['rows'] = len(df)
            data['columns'] = len(df.columns)

            # Calculate quality metrics (similar to CSV)
            # 1. Missing values (100% weight of their percentage)
            missing_values = df.isnull().sum().sum()
            total_values = data['rows'] * data['columns']
            missing_percentage = (missing_values / total_values) * 100 if total_values > 0 else 0
            missing_penalty = missing_percentage  # Full weight (100%)
            data['missing_values'] = int(missing_values)
            data['missing_percentage'] = round(missing_percentage, 2)

            # 2. Duplicate rows (50% weight of their percentage)
            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / data['rows']) * 100 if data['rows'] > 0 else 0
            duplicate_penalty = duplicate_percentage * 0.5  # 50% weight
            data['duplicate_rows'] = int(duplicate_rows)
            data['duplicate_percentage'] = round(duplicate_percentage, 2)

            # 3. Outliers in numerical columns (25% weight of their percentage)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_count = 0
            outliers_info = {}

            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_step = 1.5 * IQR
                    outliers_in_col = ((df[col] < (Q1 - outlier_step)) | (df[col] > (Q3 + outlier_step))).sum()
                    outliers_count += outliers_in_col
                    outliers_info[col] = int(outliers_in_col)

            # Safety check to avoid division by zero
            if len(numeric_cols) > 0 and data['rows'] > 0:
                outlier_percentage = (outliers_count / (len(numeric_cols) * data['rows'])) * 100
            else:
                outlier_percentage = 0

            outlier_penalty = outlier_percentage * 0.25  # 25% weight
            data['outliers'] = outliers_info
            data['outliers_count'] = int(outliers_count)
            data['outlier_percentage'] = round(outlier_percentage, 2)

            # Calculate final quality score
            total_penalty = missing_penalty + duplicate_penalty + outlier_penalty
            data['quality_score'] = round(max(0, 100 - total_penalty), 1)

            # Store quality score components for transparency
            data['quality_components'] = {
                'missing_values': {
                    'count': int(missing_values),
                    'percentage': round(missing_percentage, 2),
                    'penalty': round(missing_penalty, 2),
                    'weight': 1.0  # 100% weight
                },
                'duplicate_rows': {
                    'count': int(duplicate_rows),
                    'percentage': round(duplicate_percentage, 2),
                    'penalty': round(duplicate_penalty, 2),
                    'weight': 0.5  # 50% weight
                },
                'outliers': {
                    'count': int(outliers_count),
                    'percentage': round(outlier_percentage, 2),
                    'penalty': round(outlier_penalty, 2),
                    'weight': 0.25  # 25% weight
                }
            }

        else:
            # For other file types like images or text
            data['rows'] = 0
            data['columns'] = 0
            data['quality_score'] = 50  # Default value
            data['quality_components'] = {
                'note': 'Detailed quality analysis not available for this file type'
            }

    except Exception as e:
        # If there's an error in analysis, return basic info
        logger.error(f"Error analyzing file: {e}")
        data['rows'] = 0
        data['columns'] = 0
        data['quality_score'] = 0
        data['quality_components'] = {
            'error': str(e)
        }
    return data

# Add this class to handle PyCaret preprocessing
class PyCaretPreprocessor:
    def __init__(self, df, target=None, task_type='classification'):
        self.df = df.copy()
        self.target = target
        self.task_type = task_type
        self.setup_params = None
        self.transformed_data = None
        self.preprocessing_steps = []
        self.visualizations = []

    def analyze_dataset(self):
        """Perform initial data analysis and return insights"""
        analysis = {
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'descriptive_stats': self.df.describe().to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns}
        }
        return analysis

    def setup_pycaret(self, **kwargs):
        """Initialize PyCaret setup with custom parameters"""
        try:
            # Filter out potentially unsupported parameters
            supported_params = {}
            # Only pass through well-known supported parameters
            if 'normalize' in kwargs:
                supported_params['normalize'] = kwargs['normalize']
            if 'transformation' in kwargs:
                supported_params['transformation'] = kwargs['transformation']

            # Basic setup that should work with any PyCaret version
            if self.task_type == 'classification':
                from pycaret.classification import setup as cls_setup, pull as cls_pull
                exp = cls_setup(data=self.df, target=self.target, session_id=123,
                                verbose=False, **supported_params)
                self.setup_params = cls_pull()
            elif self.task_type == 'regression':
                from pycaret.regression import setup as reg_setup, pull as reg_pull
                exp = reg_setup(data=self.df, target=self.target, session_id=123,
                                verbose=False, **supported_params)
                self.setup_params = reg_pull()
            else:  # clustering
                from pycaret.clustering import setup as clus_setup, pull as clus_pull
                exp = clus_setup(data=self.df, session_id=123,
                                 verbose=False, **supported_params)
                self.setup_params = clus_pull()

            return self.setup_params
        except Exception as e:
            logger.error(f"PyCaret setup error: {str(e)}")
            raise Exception(f"Error in PyCaret setup: {str(e)}")

    def get_transformation_suggestions(self):
        """Get suggested preprocessing steps based on data analysis"""
        suggestions = []

        # Check for missing values
        missing_cols = [col for col, count in self.df.isnull().sum().items() if count > 0]
        if missing_cols:
            suggestions.append({
                'type': 'imputation',
                'description': f'Missing values detected in {len(missing_cols)} columns',
                'columns': missing_cols,
                'recommendation': 'Consider mean/median imputation for numerical or mode for categorical'
            })

        # Check for high cardinality categorical
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = [col for col in categorical_cols if self.df[col].nunique() > 20]
        if high_cardinality:
            suggestions.append({
                'type': 'encoding',
                'description': f'High cardinality ({len(high_cardinality)} columns with >20 unique values)',
                'columns': high_cardinality,
                'recommendation': 'Consider rare label encoding or target encoding'
            })

        # Check for numerical outliers
        numerical_cols = self.df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                suggestions.append({
                    'type': 'outlier',
                    'description': f'{outliers} outliers detected in {col}',
                    'columns': [col],
                    'recommendation': 'Consider winsorization or transformation'
                })

        return suggestions

    def apply_transformations(self, transformations):
        """Apply selected transformations and track changes"""
        self.preprocessing_steps = []
        self.visualizations = []

        # Create before snapshot
        before_stats = self.df.describe().to_dict()
        self.preprocessing_steps.append({
            'step': 'initial',
            'description': 'Original dataset',
            'changes': {},
            'stats': before_stats
        })

        # Apply each transformation
        for transform in transformations:
            step_result = self._apply_single_transformation(transform)
            if step_result:
                self.preprocessing_steps.append(step_result)

        # Get final transformed data
        if self.task_type == 'classification':
            self.transformed_data = get_config('X')
        elif self.task_type == 'regression':
            self.transformed_data = get_config('X')
        else:  # clustering
            self.transformed_data = get_config('X')

        return self.preprocessing_steps

    def _apply_single_transformation(self, transform):
        """Apply a single transformation and record its effects"""
        try:
            transform_type = transform.get('type')
            columns = transform.get('columns', [])
            params = transform.get('params', {})

            # Record before state
            before_cols = self.df[columns].copy() if columns else self.df.copy()
            before_stats = before_cols.describe().to_dict() if before_cols.select_dtypes(
                include=['number']).any().any() else {}

            # Apply transformation
            if transform_type == 'imputation':
                # PyCaret handles imputation automatically in setup
                description = f"Imputed missing values in {', '.join(columns)}"
                # We'll let PyCaret handle this in the setup phase
                return None

            elif transform_type == 'scaling':
                # This will be handled by PyCaret's normalize parameter
                description = f"Scaled features: {', '.join(columns)}"
                return None

            elif transform_type == 'encoding':
                # This will be handled by PyCaret's categorical encoding
                description = f"Encoded categorical features: {', '.join(columns)}"
                return None

            elif transform_type == 'outlier':
                # Apply winsorization
                for col in columns:
                    q1 = self.df[col].quantile(0.25)
                    q3 = self.df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    self.df[col] = self.df[col].clip(lower, upper)
                description = f"Outlier treatment applied to {', '.join(columns)}"

            elif transform_type == 'transformation':
                # Apply log transform
                for col in columns:
                    self.df[col] = np.log1p(self.df[col])
                description = f"Log transformation applied to {', '.join(columns)}"

            else:
                return None

            # Record after state
            after_cols = self.df[columns].copy() if columns else self.df.copy()
            after_stats = after_cols.describe().to_dict() if after_cols.select_dtypes(
                include=['number']).any().any() else {}

            # Generate visualization
            viz_data = self._generate_comparison_visualization(before_cols, after_cols, transform_type)

            return {
                'step': transform_type,
                'description': description,
                'changes': {
                    'before': before_stats,
                    'after': after_stats
                },
                'visualization': viz_data
            }

        except Exception as e:
            logger.error(f"Error applying transformation: {e}")
            return None

    def _generate_comparison_visualization(self, before_data, after_data, transform_type):
        """Generate before/after visualization for a transformation"""
        try:
            # Select a numerical column for visualization
            num_cols = before_data.select_dtypes(include=['number']).columns
            if len(num_cols) == 0:
                return None

            col = num_cols[0]

            # Create figure
            plt.figure(figsize=(12, 6))

            # Before plot
            plt.subplot(1, 2, 1)
            if transform_type == 'outlier':
                sns.boxplot(x=before_data[col])
            else:
                sns.histplot(before_data[col], kde=True)
            plt.title(f'Before {transform_type}')

            # After plot
            plt.subplot(1, 2, 2)
            if transform_type == 'outlier':
                sns.boxplot(x=after_data[col])
            else:
                sns.histplot(after_data[col], kde=True)
            plt.title(f'After {transform_type}')

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)

            # Encode as base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return image_base64

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None


@app.route('/preprocessing/<int:dataset_id>', methods=['GET'])
@login_required
def preprocessing_page(dataset_id):
    """Preprocessing configuration page"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        flash('Dataset not found')
        return redirect(url_for('dashboard'))

    try:
        # Read file based on type
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Sample data for preview
            sample_data = df.head(10).to_dict('records')
            columns = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]

            return render_template(
                'preprocessing.html',
                dataset=dataset,
                columns=columns,
                sample_data=sample_data
            )
        else:
            flash('Preprocessing is only available for CSV and Excel files')
            return redirect(url_for('dashboard'))

    except Exception as e:
        logger.error(f"Error loading dataset for preprocessing: {e}")
        flash(f'Error loading dataset: {str(e)}')
        return redirect(url_for('dashboard'))


@app.route('/api/preprocessing/analyze/<int:dataset_id>', methods=['POST'])
@login_required
def analyze_dataset(dataset_id):
    """Analyze dataset and return preprocessing suggestions"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        # Read file based on type
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Get task type from request or default to classification
            data = request.get_json()
            target_col = data.get('target')
            task_type = data.get('task_type', 'classification')

            # Initialize preprocessor
            preprocessor = PyCaretPreprocessor(df, target=target_col, task_type=task_type)

            # Get analysis and suggestions
            analysis = preprocessor.analyze_dataset()
            suggestions = preprocessor.get_transformation_suggestions()

            return jsonify({
                "success": True,
                "analysis": analysis,
                "suggestions": suggestions,
                "columns": list(df.columns)
            })
        else:
            return jsonify({"success": False, "message": "Unsupported file type for preprocessing"}), 400

    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return jsonify({"success": False, "message": f"Error analyzing dataset: {str(e)}"}), 500


@app.route('/api/preprocessing/transform/<int:dataset_id>', methods=['POST'])
@login_required
def transform_dataset(dataset_id):
    """Apply transformations to dataset"""
    global next_dataset_id
    user_id = session['user_id']

    logger.info(f"Starting transformation for dataset {dataset_id}")

    try:
        user_datasets = datasets.get(user_id, [])
        dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

        if not dataset:
            logger.error(f"Dataset {dataset_id} not found for user {user_id}")
            return jsonify({"success": False, "message": "Dataset not found"}), 404

        # Read file based on type
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        logger.info(f"Reading dataset from {file_path}")

        # Handle file reading errors explicitly
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                return jsonify({"success": False, "message": "Unsupported file type"}), 400
        except Exception as file_error:
            logger.error(f"Error reading file: {str(file_error)}")
            return jsonify({"success": False, "message": f"Could not read dataset file: {str(file_error)}"}), 500

        # Get transformation config
        data = request.get_json()

        # Skip PyCaret and just do basic transformations manually
        try:
            # Apply transformations directly without PyCaret
            transformed_df = df.copy()
            transformations = data.get('transformations', [])

            for transform in transformations:
                transform_type = transform.get('type')
                columns = transform.get('columns', [])

                logger.info(f"Applying {transform_type} to columns {columns}")

                if transform_type == 'imputation':
                    # Simple mean imputation for numeric columns
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(transformed_df[col]):
                            transformed_df[col] = transformed_df[col].fillna(transformed_df[col].mean())
                        else:
                            # Mode imputation for non-numeric
                            transformed_df[col] = transformed_df[col].fillna(
                                transformed_df[col].mode()[0] if not transformed_df[col].mode().empty else "")

                elif transform_type == 'scaling':
                    # Min-max scaling
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(transformed_df[col]):
                            min_val = transformed_df[col].min()
                            max_val = transformed_df[col].max()
                            if max_val > min_val:  # Avoid division by zero
                                transformed_df[col] = (transformed_df[col] - min_val) / (max_val - min_val)

                elif transform_type == 'encoding':
                    # Simple one-hot encoding
                    for col in columns:
                        if not pd.api.types.is_numeric_dtype(transformed_df[col]):
                            dummies = pd.get_dummies(transformed_df[col], prefix=col)
                            transformed_df = pd.concat([transformed_df.drop(col, axis=1), dummies], axis=1)

                elif transform_type == 'outlier':
                    # IQR-based outlier removal
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(transformed_df[col]):
                            Q1 = transformed_df[col].quantile(0.25)
                            Q3 = transformed_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            transformed_df[col] = transformed_df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

                elif transform_type == 'transformation':
                    # Log transformation
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(transformed_df[col]):
                            # Add small constant to handle zeros
                            min_val = transformed_df[col].min()
                            const = 1 if min_val >= 0 else abs(min_val) + 1
                            transformed_df[col] = np.log1p(transformed_df[col] + const)

            logger.info(f"Transformations applied successfully. Shape: {transformed_df.shape}")

        except Exception as transform_error:
            logger.error(f"Error during transformations: {str(transform_error)}")
            return jsonify(
                {"success": False, "message": f"Error applying transformations: {str(transform_error)}"}), 500

        # Save transformed data to file
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"transformed_{timestamp}_{dataset['name']}"
            new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            if file_type == 'csv':
                transformed_df.to_csv(new_path, index=False)
            else:
                transformed_df.to_excel(new_path, index=False)

            logger.info(f"Saved transformed dataset to {new_path}")
        except Exception as save_error:
            logger.error(f"Error saving transformed file: {str(save_error)}")
            return jsonify({"success": False, "message": f"Error saving transformed file: {str(save_error)}"}), 500

        # Create new dataset entry
        new_dataset_id = next_dataset_id
        next_dataset_id += 1

        new_dataset = {
            "id": new_dataset_id,
            "name": new_filename,
            "file_type": file_type,
            "file_path": new_path,
            "rows": len(transformed_df),
            "columns": len(transformed_df.columns),
            "quality_score": dataset['quality_score'],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": False,
            "is_transformed": True,
            "original_dataset_id": dataset_id
        }

        # Add to user's datasets
        if user_id not in datasets:
            datasets[user_id] = []
        datasets[user_id].append(new_dataset)

        logger.info(f"Created new dataset with ID {new_dataset_id}")

        return jsonify({
            "success": True,
            "message": "Dataset transformed successfully",
            "new_dataset_id": new_dataset_id
        })

    except Exception as e:
        logger.error(f"Unexpected error in transform_dataset: {str(e)}")
        return jsonify({"success": False, "message": f"Error transforming dataset: {str(e)}"}), 500

# todo --------------PyCaretPreprocessor (End) ----------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# ===== REQUIRED IMPORTS FOR VISUALIZATION =====
import pandas as pd
import numpy as np
import logging
from flask import jsonify, request, session, render_template, redirect, url_for, flash
from functools import wraps

# Make sure these imports are available in your backup project:
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.ensemble import IsolationForest
# from scipy import stats
# from scipy.stats import shapiro, kstest

logger = logging.getLogger(__name__)


# ===== VISUALIZATION ENDPOINTS =====

# Routes for Visualization Module
@app.route('/visualization/<int:dataset_id>')
@login_required
def visualization(dataset_id):
    """Data visualization page"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        flash('Dataset not found')
        return redirect(url_for('dashboard'))

    return render_template(
        'visualization.html',
        dataset=dataset
    )


@app.route('/api/dataset/<int:dataset_id>/info')
@login_required
def get_dataset_info(dataset_id):
    """Get dataset columns and metadata for visualization"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        # Read file to get column information
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        columns = []

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)
            columns = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]

        return jsonify({
            "success": True,
            "dataset": {
                "id": dataset["id"],
                "name": dataset["name"],
                "rows": dataset["rows"],
                "columns": dataset["columns"],
                "quality_score": dataset["quality_score"]
            },
            "columns": columns
        })

    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/data')
@login_required
def get_dataset_data(dataset_id):
    """Get dataset data for visualization"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        # Get limit parameter (default to 1000)
        limit = request.args.get('limit', 1000, type=int)

        # Read file to get data
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Limit rows for performance
            df = df.head(limit)

            # Convert to JSON-serializable format
            data = df.replace({np.nan: None}).to_dict('records')

            return jsonify(data)

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error getting dataset data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/correlation', methods=['POST'])
@login_required
def calculate_correlation(dataset_id):
    """Calculate correlation matrix for selected features"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        data = request.get_json()
        features = data.get('features', [])
        method = data.get('method', 'pearson')

        if not features or len(features) < 2:
            return jsonify({"success": False, "message": "Please select at least 2 features"}), 400

        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Check if all features exist in dataset
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                return jsonify({
                    "success": False,
                    "message": f"Features not found: {', '.join(missing_features)}"
                }), 400

            # Calculate correlation matrix
            correlation_matrix = df[features].corr(method=method).to_dict()

            return jsonify({
                "success": True,
                "correlation": correlation_matrix,
                "features": features
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/anomalies', methods=['POST'])
@login_required
def detect_anomalies(dataset_id):
    """Detect anomalies in a dataset column"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        data = request.get_json()
        feature = data.get('feature')
        method = data.get('method', 'statistical')
        sensitivity = data.get('sensitivity', 1.5)

        if not feature:
            return jsonify({"success": False, "message": "Please select a feature"}), 400

        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Check if feature exists in dataset
            if feature not in df.columns:
                return jsonify({
                    "success": False,
                    "message": f"Feature not found: {feature}"
                }), 400

            # Get numeric values for the feature
            values = df[feature].dropna().astype(float).tolist()

            # Detect anomalies based on method
            anomalies = []

            if method == 'statistical':
                # IQR method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - sensitivity * iqr
                upper_bound = q3 + sensitivity * iqr

                for i, val in enumerate(values):
                    if val < lower_bound or val > upper_bound:
                        anomalies.append({"index": i, "value": val})

            elif method == 'zscore':
                # Z-score method
                mean = np.mean(values)
                std = np.std(values)

                for i, val in enumerate(values):
                    z_score = abs((val - mean) / std) if std > 0 else 0
                    if z_score > sensitivity:
                        anomalies.append({"index": i, "value": val})

            elif method == 'isolation':
                # Simple isolation forest (using sklearn)
                try:
                    from sklearn.ensemble import IsolationForest

                    # Reshape for sklearn
                    X = np.array(values).reshape(-1, 1)

                    # Adjust contamination based on sensitivity
                    contamination = min(0.1, max(0.01, 1.0 / sensitivity))

                    # Train isolation forest
                    clf = IsolationForest(contamination=contamination, random_state=42)
                    outlier_labels = clf.fit_predict(X)

                    # Extract anomalies (outlier_labels == -1)
                    for i, (val, label) in enumerate(zip(values, outlier_labels)):
                        if label == -1:
                            anomalies.append({"index": i, "value": val})

                except ImportError:
                    # Fallback to statistical method if sklearn not available
                    logger.warning("sklearn not available, falling back to statistical method")
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - sensitivity * iqr
                    upper_bound = q3 + sensitivity * iqr

                    for i, val in enumerate(values):
                        if val < lower_bound or val > upper_bound:
                            anomalies.append({"index": i, "value": val})

            # Calculate statistics
            stats = {
                "count": len(values),
                "min": float(min(values)),
                "max": float(max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "anomalies_count": len(anomalies),
                "anomalies_percentage": round(len(anomalies) / len(values) * 100, 2) if values else 0
            }

            return jsonify({
                "success": True,
                "anomalies": anomalies,
                "stats": stats,
                "feature": feature
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


# EDA API Endpoints
@app.route('/api/dataset/<int:dataset_id>/eda/univariate', methods=['POST'])
@login_required
def eda_univariate(dataset_id):
    """Generate univariate analysis for a feature"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        data = request.get_json()
        feature = data.get('feature')
        plot_types = data.get('plot_types', ['histogram', 'boxplot'])

        if not feature:
            return jsonify({"success": False, "message": "Please select a feature"}), 400

        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Check if feature exists
            if feature not in df.columns:
                return jsonify({"success": False, "message": f"Feature not found: {feature}"}), 400

            # Get feature data
            feature_data = df[feature].dropna()

            # Determine feature type
            is_numeric = pd.api.types.is_numeric_dtype(feature_data)
            is_categorical = pd.api.types.is_categorical_dtype(feature_data) or feature_data.nunique() < 20

            # Generate summary statistics
            summary = {}
            if is_numeric:
                summary = {
                    "count": int(feature_data.count()),
                    "mean": float(feature_data.mean()) if not feature_data.empty else None,
                    "std": float(feature_data.std()) if not feature_data.empty else None,
                    "min": float(feature_data.min()) if not feature_data.empty else None,
                    "25%": float(feature_data.quantile(0.25)) if not feature_data.empty else None,
                    "50%": float(feature_data.quantile(0.5)) if not feature_data.empty else None,
                    "75%": float(feature_data.quantile(0.75)) if not feature_data.empty else None,
                    "max": float(feature_data.max()) if not feature_data.empty else None,
                    "skew": float(feature_data.skew()) if not feature_data.empty else None,
                    "kurtosis": float(feature_data.kurtosis()) if not feature_data.empty else None
                }
            else:
                # For categorical, calculate frequencies
                value_counts = feature_data.value_counts().to_dict()
                summary = {
                    "count": int(feature_data.count()),
                    "unique_values": int(feature_data.nunique()),
                    "most_common": feature_data.mode()[0] if not feature_data.empty else None,
                    "frequencies": value_counts
                }

            # Generate insights based on statistics
            insights = []
            if is_numeric:
                mean = summary.get("mean")
                std = summary.get("std")
                skew = summary.get("skew")
                kurtosis = summary.get("kurtosis")

                if abs(skew) > 1:
                    skew_direction = "positively" if skew > 0 else "negatively"
                    insights.append(f"The distribution is {skew_direction} skewed (skewness: {skew:.2f}).")

                if kurtosis > 1:
                    insights.append(f"The distribution has heavy tails (kurtosis: {kurtosis:.2f}).")
                elif kurtosis < -1:
                    insights.append(f"The distribution has light tails (kurtosis: {kurtosis:.2f}).")

                iqr = summary.get("75%") - summary.get("25%")
                if iqr > 0:
                    potential_outliers = feature_data[(feature_data < summary.get("25%") - 1.5 * iqr) |
                                                      (feature_data > summary.get("75%") + 1.5 * iqr)]
                    outlier_count = len(potential_outliers)
                    if outlier_count > 0:
                        insights.append(f"There are {outlier_count} potential outliers in the data.")
            else:
                # Insights for categorical data
                most_common = summary.get("most_common")
                unique_values = summary.get("unique_values")
                frequencies = summary.get("frequencies", {})

                if most_common and frequencies:
                    most_common_count = frequencies.get(most_common, 0)
                    most_common_pct = (most_common_count / summary.get("count")) * 100
                    insights.append(f"The most common value is '{most_common}' ({most_common_pct:.1f}% of data).")

                if unique_values:
                    insights.append(f"There are {unique_values} unique values in this feature.")

            # Return results
            return jsonify({
                "success": True,
                "feature": feature,
                "is_numeric": is_numeric,
                "is_categorical": is_categorical,
                "summary": summary,
                "insights": insights,
                "values": feature_data.to_list()[:1000]  # Limit number of values sent to browser
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error in univariate analysis: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/eda/bivariate', methods=['POST'])
@login_required
def eda_bivariate(dataset_id):
    """Generate bivariate analysis for two features"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        data = request.get_json()
        feature1 = data.get('feature1')
        feature2 = data.get('feature2')
        plot_types = data.get('plot_types', ['scatter', 'heatmap'])

        if not feature1 or not feature2:
            return jsonify({"success": False, "message": "Please select both features"}), 400

        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Check if features exist
            if feature1 not in df.columns:
                return jsonify({"success": False, "message": f"Feature not found: {feature1}"}), 400
            if feature2 not in df.columns:
                return jsonify({"success": False, "message": f"Feature not found: {feature2}"}), 400

            # Drop rows with missing values in either feature
            df_clean = df[[feature1, feature2]].dropna()

            # Determine feature types
            is_numeric1 = pd.api.types.is_numeric_dtype(df_clean[feature1])
            is_numeric2 = pd.api.types.is_numeric_dtype(df_clean[feature2])
            is_categorical1 = pd.api.types.is_categorical_dtype(df_clean[feature1]) or df_clean[feature1].nunique() < 20
            is_categorical2 = pd.api.types.is_categorical_dtype(df_clean[feature2]) or df_clean[feature2].nunique() < 20

            # Calculate relationship metrics based on feature types
            relationship_metrics = {}
            insights = []

            if is_numeric1 and is_numeric2:
                # Correlation for two numeric features
                correlation = df_clean[feature1].corr(df_clean[feature2])
                relationship_metrics["correlation"] = float(correlation)

                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"

                direction = "positive" if correlation > 0 else "negative"

                insights.append(
                    f"There is a {strength} {direction} correlation between {feature1} and {feature2} (r = {correlation:.2f}).")

            elif (is_numeric1 and is_categorical2) or (is_numeric2 and is_categorical1):
                # ANOVA-like analysis for numeric vs categorical
                numeric_feature = feature1 if is_numeric1 else feature2
                categorical_feature = feature2 if is_numeric1 else feature1

                # Group by categorical and calculate stats for numeric
                grouped_stats = df_clean.groupby(categorical_feature)[numeric_feature].agg(
                    ['mean', 'std', 'count']).to_dict()
                relationship_metrics["grouped_stats"] = grouped_stats

                # Calculate F statistic (simplified)
                categories = df_clean[categorical_feature].unique()
                if len(categories) > 1:
                    group_means = [df_clean[df_clean[categorical_feature] == cat][numeric_feature].mean() for cat in
                                   categories]
                    overall_mean = df_clean[numeric_feature].mean()

                    # Simple measure of between-group variation
                    between_group_var = sum([(gm - overall_mean) ** 2 for gm in group_means]) / len(group_means)
                    within_group_var = df_clean[numeric_feature].var()

                    f_ratio = between_group_var / within_group_var if within_group_var > 0 else 0
                    relationship_metrics["f_ratio"] = float(f_ratio)

                    if f_ratio > 10:
                        insights.append(
                            f"There are significant differences in {numeric_feature} across different {categorical_feature} groups.")
                    elif f_ratio > 1:
                        insights.append(
                            f"There are some differences in {numeric_feature} across different {categorical_feature} groups.")

            elif is_categorical1 and is_categorical2:
                # Contingency table and chi-square for two categorical features
                contingency_table = pd.crosstab(df_clean[feature1], df_clean[feature2])
                relationship_metrics["contingency_table"] = contingency_table.to_dict()

                # Calculate Cramer's V (association measure for categorical variables)
                chi2 = 0
                n = contingency_table.sum().sum()

                if n > 0:
                    # Expected frequencies under independence
                    row_sums = contingency_table.sum(axis=1)
                    col_sums = contingency_table.sum(axis=0)

                    for i, row in enumerate(contingency_table.index):
                        for j, col in enumerate(contingency_table.columns):
                            observed = contingency_table.iloc[i, j]
                            expected = row_sums[row] * col_sums[col] / n

                            if expected > 0:
                                chi2 += ((observed - expected) ** 2) / expected

                    # Cramer's V
                    r = len(contingency_table.index)
                    c = len(contingency_table.columns)
                    cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1))) if min(r - 1, c - 1) > 0 else 0

                    relationship_metrics["chi2"] = float(chi2)
                    relationship_metrics["cramers_v"] = float(cramers_v)

                    if cramers_v > 0.5:
                        insights.append(f"There is a strong association between {feature1} and {feature2}.")
                    elif cramers_v > 0.3:
                        insights.append(f"There is a moderate association between {feature1} and {feature2}.")
                    else:
                        insights.append(f"There is a weak association between {feature1} and {feature2}.")

            # Return results
            return jsonify({
                "success": True,
                "feature1": feature1,
                "feature2": feature2,
                "is_numeric1": is_numeric1,
                "is_numeric2": is_numeric2,
                "is_categorical1": is_categorical1,
                "is_categorical2": is_categorical2,
                "relationship_metrics": relationship_metrics,
                "insights": insights,
                "data": df_clean.to_dict(orient='records')[:1000]  # Limit to 1000 records
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error in bivariate analysis: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/eda/multivariate', methods=['POST'])
@login_required
def eda_multivariate(dataset_id):
    """Generate multivariate analysis for multiple features"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        data = request.get_json()
        features = data.get('features', [])
        plot_types = data.get('plot_types', ['pairplot'])

        if not features or len(features) < 2:
            return jsonify({"success": False, "message": "Please select at least 2 features"}), 400

        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Check if features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                return jsonify({
                    "success": False,
                    "message": f"Features not found: {', '.join(missing_features)}"
                }), 400

            # Drop rows with missing values
            df_clean = df[features].dropna()

            # Determine feature types
            feature_types = {}
            for feature in features:
                feature_types[feature] = {
                    "is_numeric": pd.api.types.is_numeric_dtype(df_clean[feature]),
                    "is_categorical": pd.api.types.is_categorical_dtype(df_clean[feature]) or df_clean[
                        feature].nunique() < 20
                }

            # Calculate correlation matrix
            correlation_matrix = {}
            numeric_features = [f for f in features if feature_types[f]["is_numeric"]]

            if len(numeric_features) >= 2:
                correlation_matrix = df_clean[numeric_features].corr().to_dict()

            # Generate insights based on correlations
            insights = []

            # Find strong correlations
            if correlation_matrix:
                strong_correlations = []
                for i, f1 in enumerate(numeric_features):
                    for j, f2 in enumerate(numeric_features):
                        if i < j:  # Only look at unique pairs
                            corr = correlation_matrix.get(f1, {}).get(f2, 0)
                            if abs(corr) > 0.7:
                                strong_correlations.append({
                                    "feature1": f1,
                                    "feature2": f2,
                                    "correlation": corr
                                })

                strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

                for i, corr in enumerate(strong_correlations[:3]):  # Top 3 correlations
                    direction = "positive" if corr["correlation"] > 0 else "negative"
                    insights.append(
                        f"Strong {direction} correlation between {corr['feature1']} and {corr['feature2']} (r = {corr['correlation']:.2f}).")

            # If PCA is requested, calculate principal components
            pca_results = None
            if 'pcaCheck' in plot_types and len(numeric_features) >= 2:
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA

                    # Scale data
                    X = df_clean[numeric_features]
                    X_scaled = StandardScaler().fit_transform(X)

                    # Calculate PCA with 2 components for visualization
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(X_scaled)

                    pca_results = {
                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                        "components": pca.components_.tolist(),
                        "transformed_data": principal_components.tolist()
                    }

                    # Add PCA insights
                    total_variance = sum(pca.explained_variance_ratio_) * 100
                    insights.append(
                        f"The first two principal components explain {total_variance:.1f}% of the total variance.")

                    if total_variance > 80:
                        insights.append("This suggests strong structure and correlation in the data.")
                    elif total_variance < 40:
                        insights.append("This suggests weak structure and limited correlation in the data.")

                except ImportError:
                    logger.warning("sklearn not available for PCA analysis")

            # Return results
            return jsonify({
                "success": True,
                "features": features,
                "feature_types": feature_types,
                "correlation_matrix": correlation_matrix,
                "insights": insights,
                "pca_results": pca_results,
                "data": df_clean.to_dict(orient='records')[:1000]  # Limit to 1000 records
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error in multivariate analysis: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/eda/missing', methods=['POST'])
@login_required
def eda_missing_values(dataset_id):
    """Analyze missing values in the dataset"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Calculate missing value statistics
            total_cells = df.shape[0] * df.shape[1]
            total_missing = df.isna().sum().sum()
            missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0

            # Missing values by column
            missing_by_column = {}
            for column in df.columns:
                missing_count = df[column].isna().sum()
                missing_by_column[column] = {
                    "count": int(missing_count),
                    "percentage": float((missing_count / df.shape[0]) * 100) if df.shape[0] > 0 else 0
                }

            # Missing values by row (sample of rows with most missing values)
            missing_by_row = {}
            row_missing_counts = df.isna().sum(axis=1)
            most_missing_rows = row_missing_counts.nlargest(10).index.tolist()

            for row_idx in most_missing_rows:
                missing_count = row_missing_counts[row_idx]
                missing_by_row[int(row_idx)] = {
                    "count": int(missing_count),
                    "percentage": float((missing_count / df.shape[1]) * 100) if df.shape[1] > 0 else 0
                }

            # Generate insights about missing data
            insights = []

            if missing_percentage == 0:
                insights.append("There are no missing values in the dataset.")
            else:
                insights.append(
                    f"Overall, {missing_percentage:.1f}% of the data is missing ({total_missing} of {total_cells} cells).")

                # Find columns with high missing rates
                high_missing_columns = [col for col, stats in missing_by_column.items() if stats["percentage"] > 20]
                if high_missing_columns:
                    if len(high_missing_columns) <= 3:
                        columns_list = ", ".join(high_missing_columns)
                        insights.append(f"Columns with high missing rates: {columns_list}")
                    else:
                        insights.append(f"{len(high_missing_columns)} columns have more than 20% missing values.")

                # Patterns in missing data
                if df.shape[1] >= 2:
                    # Simple analysis of potential patterns by checking correlation of missingness
                    missing_indicators = df.isna().astype(int)

                    # Find pairs of columns with correlated missingness
                    correlated_missing = []

                    for i, col1 in enumerate(df.columns[:-1]):
                        for col2 in df.columns[i + 1:]:
                            if missing_by_column[col1]["count"] > 0 and missing_by_column[col2]["count"] > 0:
                                # Correlation between missing indicators
                                corr = missing_indicators[col1].corr(missing_indicators[col2])
                                if abs(corr) > 0.7:
                                    correlated_missing.append({
                                        "column1": col1,
                                        "column2": col2,
                                        "correlation": float(corr)
                                    })

                    if correlated_missing:
                        insights.append(
                            f"Found {len(correlated_missing)} pairs of columns with correlated missing values, suggesting systematic patterns in missingness.")

            # Return results
            return jsonify({
                "success": True,
                "total_cells": total_cells,
                "total_missing": int(total_missing),
                "missing_percentage": float(missing_percentage),
                "missing_by_column": missing_by_column,
                "missing_by_row": missing_by_row,
                "insights": insights
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error in missing values analysis: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/dataset/<int:dataset_id>/eda/distribution', methods=['POST'])
@login_required
def eda_distribution_comparison(dataset_id):
    """Compare distribution of a feature to theoretical distributions"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404

    try:
        data = request.get_json()
        feature = data.get('feature')
        dist_type = data.get('dist_type', 'normal')

        if not feature:
            return jsonify({"success": False, "message": "Please select a feature"}), 400

        # Read file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)

            # Check if feature exists
            if feature not in df.columns:
                return jsonify({"success": False, "message": f"Feature not found: {feature}"}), 400

            # Get feature data
            feature_data = df[feature].dropna()

            # Check if feature is numeric
            if not pd.api.types.is_numeric_dtype(feature_data):
                return jsonify(
                    {"success": False, "message": "Feature must be numeric for distribution comparison"}), 400

            # Basic statistics of the feature
            mean = float(feature_data.mean())
            std = float(feature_data.std())
            min_val = float(feature_data.min())
            max_val = float(feature_data.max())

            # Generate theoretical distribution points
            try:
                from scipy import stats

                x = np.linspace(min_val, max_val, 100)
                theoretical_dist = None
                normality_test = None

                if dist_type == 'normal':
                    y = stats.norm.pdf(x, loc=mean, scale=std)
                    theoretical_dist = {"x": x.tolist(), "y": y.tolist(), "name": "Normal Distribution"}

                    # Conduct normality test
                    try:
                        from scipy.stats import shapiro
                        stat, p_value = shapiro(feature_data)
                        normality_test = {"test": "Shapiro-Wilk", "statistic": float(stat), "p_value": float(p_value)}
                    except ImportError:
                        logger.warning("scipy.stats not available for normality test")

                elif dist_type == 'uniform':
                    y = stats.uniform.pdf(x, loc=min_val, scale=max_val - min_val)
                    theoretical_dist = {"x": x.tolist(), "y": y.tolist(), "name": "Uniform Distribution"}

                    # Conduct uniformity test (K-S test against uniform distribution)
                    try:
                        from scipy.stats import kstest
                        stat, p_value = kstest(feature_data, 'uniform', args=(min_val, max_val - min_val))
                        normality_test = {"test": "Kolmogorov-Smirnov", "statistic": float(stat),
                                          "p_value": float(p_value)}
                    except ImportError:
                        logger.warning("scipy.stats not available for uniformity test")

                elif dist_type == 'exponential':
                    # For exponential, use rate = 1/mean
                    rate = 1 / mean if mean > 0 else 1
                    y = stats.expon.pdf(x, scale=1 / rate)
                    theoretical_dist = {"x": x.tolist(), "y": y.tolist(), "name": "Exponential Distribution"}

                    # No specific test for exponential, use K-S test
                    try:
                        from scipy.stats import kstest
                        stat, p_value = kstest(feature_data, 'expon', args=(0, mean))
                        normality_test = {"test": "Kolmogorov-Smirnov", "statistic": float(stat),
                                          "p_value": float(p_value)}
                    except ImportError:
                        logger.warning("scipy.stats not available for exponential test")

                elif dist_type == 'lognormal':
                    # For lognormal, estimate parameters
                    if (feature_data > 0).all():
                        log_data = np.log(feature_data)
                        log_mean = log_data.mean()
                        log_std = log_data.std()
                        y = stats.lognorm.pdf(x, s=log_std, scale=np.exp(log_mean))
                        theoretical_dist = {"x": x.tolist(), "y": y.tolist(), "name": "Log-Normal Distribution"}

                        # Test lognormality by testing if log-transformed data is normal
                        try:
                            from scipy.stats import shapiro
                            stat, p_value = shapiro(log_data)
                            normality_test = {"test": "Shapiro-Wilk (log-transformed)", "statistic": float(stat),
                                              "p_value": float(p_value)}
                        except ImportError:
                            logger.warning("scipy.stats not available for lognormal test")
                    else:
                        return jsonify(
                            {"success": False, "message": "Feature must be positive for log-normal distribution"}), 400

            except ImportError:
                logger.warning("scipy not available for distribution analysis")
                theoretical_dist = None
                normality_test = None

            # Generate histogram bins for actual data
            hist, bin_edges = np.histogram(feature_data, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Generate insights
            insights = []

            if normality_test:
                significance = 0.05
                if normality_test["p_value"] > significance:
                    insights.append(
                        f"The data is consistent with a {dist_type} distribution (p-value: {normality_test['p_value']:.4f}).")
                else:
                    insights.append(
                        f"The data is significantly different from a {dist_type} distribution (p-value: {normality_test['p_value']:.4f}).")

            # Return results
            return jsonify({
                "success": True,
                "feature": feature,
                "histogram": {
                    "x": bin_centers.tolist(),
                    "y": hist.tolist(),
                    "name": "Actual Data"
                },
                "theoretical_dist": theoretical_dist,
                "statistics": {
                    "mean": mean,
                    "std": std,
                    "min": min_val,
                    "max": max_val
                },
                "normality_test": normality_test,
                "insights": insights
            })

        return jsonify({"success": False, "message": "Unsupported file type"}), 400

    except Exception as e:
        logger.error(f"Error in distribution comparison: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

