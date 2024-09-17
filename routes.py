from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import string
import yaml
import random
import re
import yfinance as yf
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from models import create_model
from tables import db, User, Symbol, TrainedModels, TemporaryPassword

# Read the configuration file at the top of the script
def read_settings(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the settings file
settings = read_settings('settings.yaml')

def generate_temporary_password(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def send_email(receiver_email, temporary_password,
               mail_type = 'Reset'):
    # Placeholder function to simulate sending an email
    # Implement actual email sending logic here
    print(f"Sending temporary password to {receiver_email}: {temporary_password}")

    #Sender Info
    sender_email = settings['email']['sender_email']
    sender_password = settings['email']['sender_password']
    smtp_server = settings['email']['smtp_server']
    smtp_port = settings['email']['smtp_port']
    #Message Info
    subject = ''
    body = ''
    if mail_type == 'Reset':
        subject = 'Your Temporary Password for AiDea Account Recovery'
        body = (
            f"Dear User,\n\n"
            f"We have generated a temporary password for you to reset your account password:\n\n"
            f"Temporary Password: {temporary_password}\n\n"
            f"For your security, please do not share this password with anyone. "
            f"If you did not request this reset, please contact our support team immediately.\n\n"
            f"Thank you for your attention.\n\n"
            f"Best regards,\n\n"
            f"AiDea Support Team")
    elif mail_type =='Sign-Up':
        subject = 'Your Temporary Password for AiDea Sign-Up'
        body = (
            f"Dear User,\n\n"
            f"We have generated a temporary password for you to sign-up:\n\n"
            f"Temporary Password: {temporary_password}\n\n"
            f"For your security, please do not share this password with anyone. "
            f"If you did not request to sign-up, please ignore this message.\n\n"
            f"Thank you for your attention.\n\n"
            f"Best regards,\n\n"
            f"AiDea Support Team")
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    #Message Sending
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
    finally:
        server.quit()

def is_valid_password(password):
    """Check if the password is valid based on the criteria."""
    if len(password) < 8: # Check password length
        return False
    if not re.search(r'[A-Z]', password):  # Check for uppercase letter
        return False
    if not re.search(r'[a-z]', password):  # Check for lowercase letter
        return False
    if not re.search(r'[0-9]', password):  # Check for number
        return False
    return True

def convert_to_days(input_length):
    days = 0
    if input_length == '1_week':
        days = 7
    elif input_length == '1_month':
        days = 30
    elif input_length == '2_months':
        days = 60
    elif input_length == '3_months':
        days = 90
    elif input_length == '4_months':
        days = 120
    elif input_length == '5_months':
        days = 150
    elif input_length == '6_months':
        days = 180
    elif input_length == '9_months':
        days = 270
    elif input_length == '1_year':
        days = 360
    elif input_length == '2_years':
        days = 720
    elif input_length == '3_years':
        days = 1080
    elif input_length == '4_years':
        days = 1440
    elif input_length == '5_years':
        days = 1800
        
    return days

def fetch_data(symbol, start_date, end_date): # fetch data based on user preference
    data = None
    try:
        # Fetch historical price data
        df = yf.download(symbol, start=start_date, end=end_date)

        # Drop 'Adj Close' column if present
        if 'Adj Close' in df.columns:
            df.drop(columns=['Adj Close'], inplace=True)

        # Store the dataframe with technical indicators
        data = df

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

    return data

def check_symbol_existence(symbol): #check the existence of symbol from yahoo finance
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        return not data.empty
    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {e}")
        return False

def init_routes(app):
    @app.route('/')
    def index():
        return render_template('login.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():

        if request.method == 'GET':
            # Serve the signup and forgot password page
            return render_template('login.html')
        
        if request.method == 'POST':
            data = request.json
            email = data.get('email')
            password = data.get('password')
            
            user = User.query.filter_by(email=email).first()

            if user and check_password_hash(user.password, password):
                # Store user ID in session
                session['user_id'] = user.id   # Store the id in the session
                session['email'] = user.email  # Store the email in the session
                session['account_type'] = user.account_type # Store the account type in the session
                return jsonify({'message': 'Login successful!', 'redirect': url_for('main'), 'account_type': user.account_type}), 200
            else:
                return jsonify({'message': 'Invalid credentials'}), 401
            
    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'GET':
            # Serve the signup page
            return render_template('signup.html')
        
        if request.method == 'POST':
            data = request.json
            if not data:
                return jsonify({'message': 'No data provided'}), 400
            
            email = data.get('email')
            password = data.get('password')
            password_re = data.get('password_re')
            
            # Check if all fields are present
            if not email or not password or not password_re:
                return jsonify({'message': 'All fields are required.'}), 400

            # Check if email already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                return jsonify({'message': 'User already exists.'}), 400

            # Check if passwords match
            if password != password_re:
                return jsonify({'message': 'Passwords do not match.'}), 400
            
            # Validate password strength
            if not is_valid_password(password):
                return jsonify({'message': 'Password must be at least 8 characters long, include uppercase letters, lowercase letters, and numbers.'}), 400
            
            # Create new user
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            session['signupEmail'] = email
            session['signupHashedPassword'] = hashed_password

            # Generate a temporary password
            temp_password = generate_temporary_password()
            # Check if a temporary password entry already exists for the email
            temp_password_entry = TemporaryPassword.query.filter_by(email=email).first()
            if temp_password_entry: # If so, delete the previous one
                db.session.delete(temp_password_entry)
                db.session.commit()

            #Replace it with the new one
            temp_password_entry = TemporaryPassword(email=email, temp_password=temp_password) 
            db.session.add(temp_password_entry)
            db.session.commit()
            
            # Send a temporary password
            send_email(email, temp_password, mail_type= 'Sign-Up')
            

            return jsonify({'message': 'A temporary password for account verification is sent to your email', 'redirect': url_for('account_verification')}), 200

    @app.route('/account_verification', methods=['GET', 'POST'])
    def account_verification():
        if request.method == 'GET':
            return render_template('account_verification.html')
        
        if request.method == 'POST':
            data = request.get_json()
            temp_password = data.get('tempPassword')
            email = session.get('signupEmail')
            hashed_password = session.get('signupHashedPassword')
            
            #Check if the entered temporary password is correct
            temp_password_entry = TemporaryPassword.query.filter_by(email=email, temp_password=temp_password).first()
            if temp_password_entry:
                # Temporary password matches
                new_user = User(email=email, password=hashed_password, account_type='basic')
                try:
                    db.session.add(new_user)
                    db.session.commit()
                    return jsonify({'message': 'Sign-up successful!', 'redirect': url_for('login')}), 200
                except:
                    return jsonify({'message': 'Error creating user.'}), 500
            else:
                return jsonify({'success': False, 'message': 'Invalid temporary password'})

    @app.route('/request_email', methods=['GET', 'POST'])
    def request_email():
        if request.method == 'POST':
            data = request.get_json()
            email = data.get('email')
            
            # Check if the email exists in the User database
            user = User.query.filter_by(email=email).first()
            if user:  
                # Generate a temporary password
                temp_password = generate_temporary_password()
                # Check if a temporary password entry already exists for the email
                temp_password_entry = TemporaryPassword.query.filter_by(email=email).first()
                if temp_password_entry: # If so, delete the previous one
                    db.session.delete(temp_password_entry)
                    db.session.commit()

                #Replace it with the new one
                temp_password_entry = TemporaryPassword(email=email, temp_password=temp_password) 
                db.session.add(temp_password_entry)
                db.session.commit()
                
                # Send a temporary password
                send_email(email, temp_password)

                # Store the email in a session to use in forgot password section
                session['resetEmail'] = email
                
                return jsonify({'success': True, 'message': 'Temporary password sent to your email'})
            else:
                return jsonify({'success': False, 'message': 'Email not found'})

        return render_template('request_email.html')

    @app.route('/forgot_password', methods=['GET', 'POST'])
    def forgot_password():
        if request.method == 'POST':
            data = request.get_json()
            temp_password = data.get('tempPassword')
            
            #Retrieve email from session
            email = session.get('resetEmail')

            if not email:
                return jsonify({'success': False, 'message': 'No email found in session. Please request an email first.'})

            # Check the temporary password in the database
            temp_password_entry = TemporaryPassword.query.filter_by(email=email, temp_password=temp_password).first()

            
            if temp_password_entry:
                # Temporary password matches
                return jsonify({'success': True, 'message': 'Temporary password verified. Proceed to reset password.'})
            else:
                return jsonify({'success': False, 'message': 'Invalid temporary password'})

        return render_template('forgot_password.html')

    @app.route('/reset_password', methods=['GET', 'POST'])
    def reset_password():
        if request.method == 'POST':
            data = request.get_json()
            new_password = data.get('newPassword')
            new_password_re = data.get('newPassword_re')

            # Retrieve the email from session
            email = session.get('resetEmail')

            if not email:
                return jsonify({'success': False, 'message': 'No email found in session. Please request an email first.'})

            if new_password != new_password_re:
                return jsonify({'success': False, 'message': 'Passwords do not match'})
            
            # Validate password strength
            if not is_valid_password(new_password):
                return jsonify({'message': 'Password must be at least 8 characters long, include uppercase letters, lowercase letters, and numbers.'}), 400

            # Fetch the user from the database
            user = User.query.filter_by(email=email).first()
            if user:
                # Check if the new password is different from the current password
                if check_password_hash(user.password, new_password):
                    return jsonify({'success': False, 'message': 'New password must be different from the old password'})
                
                # Update the password
                hashed_new_password = generate_password_hash(new_password, method='pbkdf2:sha256')
                user.password = hashed_new_password
                db.session.commit()
                return jsonify({'success': True, 'message': 'Password successfully updated. You can now login with your new password.'})
            else:
                return jsonify({'success': False, 'message': 'User not found'})

        # Ensure session email is set, otherwise redirect to request_email
        if not session.get('resetEmail'):
            return jsonify({'redirect': '/request_email', 'message': 'No email found in session. Please request an email first.'})

        return render_template('reset_password.html')
    
    @app.route('/get_user_account_type')
    def get_user_account_type():
        user_id = session.get('user_id')
        if user_id:
            user = User.query.get(user_id)
            if user:
                return jsonify({'account_type': user.account_type})
        return jsonify({'account_type': 'Unknown'}), 400


    @app.route('/forecaster', methods=['GET', 'POST']) # Forecaster Button 
    def main():
        # Data fetching
        return render_template('main.html')

    @app.route('/logout', methods=['GET', 'POST']) # Logout Button
    def logout():
        # Perform logout operations if needed
        session.pop('user_id', None)  # Remove user ID from session
        session.pop('email', None)  # Remove user ID from session
        session.pop('account_type', None)  # Remove account type from session
        return redirect(url_for('login'))

    @app.route('/my_profile', methods=['GET'])
    def my_profile():
        user_id = session['user_id']
        user = User.query.get(user_id)  # Fetch the user by ID
        
        return render_template('my_profile.html', user=user)

    @app.route('/about_models', methods=['GET', 'POST']) # About Models Button
    def about_models():
        return render_template('about_models.html')

    @app.route('/get_symbols', methods=['GET']) # Getter method to fetch symbols from the users account
    def get_symbols():
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'User not logged in'}), 401

        symbols = Symbol.query.filter_by(user_id=user_id).all()
        symbol_list = [symbol.name for symbol in symbols]

        return jsonify({'symbols': symbol_list}), 200
        
    @app.route('/add_symbol', methods=['POST'])
    def add_symbol(): # add symbol to the db
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'User not logged in'}), 401

        data = request.json
        symbol_name = data.get('symbol')
        if not symbol_name:
            return jsonify({'success': False, 'message': 'Symbol name not provided'}), 400

        if not check_symbol_existence(symbol_name):
            return jsonify({'success': False, 'message': 'Symbol does not exist'}), 404

        existing_symbol = Symbol.query.filter_by(name=symbol_name, user_id=user_id).first()
        if existing_symbol:
            return jsonify({'success': False, 'message': 'Symbol already added'}), 400
        
        new_symbol = Symbol(name=symbol_name, user_id=user_id)
        db.session.add(new_symbol)
        db.session.commit()

        return jsonify({'success': True}), 200

    @app.route('/delete_symbol', methods=['POST'])
    def delete_symbol(): # delete symbol from the db
        data = request.json
        if not data or 'symbol' not in data:
            return jsonify({'message': 'Invalid request'}), 400

        symbol_name = data['symbol']
        user_id = session.get('user_id')

        # Find and delete the symbol
        symbol = Symbol.query.filter_by(name=symbol_name, user_id=user_id).first()
        if symbol:
            db.session.delete(symbol)
            db.session.commit()
            return jsonify({'success': True}), 200
        else:
            return jsonify({'message': 'Symbol not found'}), 404
        

    @app.route('/predict', methods=['GET', 'POST'])
    def predict(): # predict button
        try:
            data = request.json
            symbol = data['symbol']
            data_length = convert_to_days(data['data_length'])
            forecast_days = convert_to_days(data['forecast_days'])
            model_type = data['model_type']
            #Set start and end date
            now = datetime.now()
            start_date =  (now - timedelta(days = data_length)).strftime("%Y-%m-%d")
            end_date = now.strftime("%Y-%m-%d")
            #Check if model already exists
            model_obj = None
            if model_type != 'AI': # Model is not AI
                model_obj = TrainedModels.load_trained_model(model_type=model_type, start_date=start_date, end_date=end_date, symbol=symbol)
                if not model_obj:  # if the model is already trained avoid re-training it.
                    data = fetch_data(symbol, start_date, end_date)
                    model_obj = create_model(model_type, data, symbol)
                    model_obj.train()
                    # Create a new instance of Model
                    new_model = TrainedModels(symbol= symbol, model_type= model_type, start_date= start_date, end_date= end_date)
                    # Save the trained model to the database
                    new_model.save_trained_model(model_obj)
            else: # Model is AI
                data = fetch_data(symbol, start_date, end_date)
                model_obj = create_model(model_type, data, symbol)
                model_obj.train()

            # Forecast
            plot_data = model_obj.forecast(forecast_days=forecast_days)
            return jsonify({'plot': plot_data}), 200

        except KeyError as e:
            return jsonify({'error': f'Missing key: {e.args[0]}'}, 400)

        except Exception as e:
            return jsonify({'error': str(e)}, 500)
        
    @app.route('/delete_account', methods=['POST'])
    def delete_account():
        if 'user_id' not in session:
            #flash('You need to log in first.', 'warning')
            return redirect(url_for('login'))

        email = session['email']
        user_id = session['user_id']
        user = User.query.get(user_id)  # Fetch user by ID
  
        if user:
            # Delete from temporary password entries
            temporary_password_entry = TemporaryPassword.query.filter_by(email=email).first()
            db.session.delete(temporary_password_entry)
            db.session.commit()
            # Delete from users table hence the symbols table
            Symbol.query.filter_by(user_id=user.id).delete()
            db.session.delete(user)
            db.session.commit()

            #Remove from the session
            session.pop('user_id', None)
            session.pop('email', None)  

            return redirect(url_for('login'))  # Redirect to login page
        else:
            #flash('Account deletion failed.', 'danger')
            return redirect(url_for('my_profile'))
        
    @app.route('/upgrade_to_premium', methods=['POST'])
    def upgrade_to_premium():
        if 'user_id' not in session:
            return redirect(url_for('login'))

        user_id = session['user_id']
        user = User.query.get(user_id)  # Fetch user by ID

        if user and user.account_type == 'basic':
            user.account_type = 'premium'
            db.session.commit()
            print('Upgrade completed')
            return '', 204  # Return no content
        else:
            return '', 400  # Bad request if upgrade fails

    @app.route('/downgrade_to_basic', methods=['POST'])
    def downgrade_to_basic():
        if 'user_id' not in session:
            return redirect(url_for('login'))

        user_id = session['user_id']
        user = User.query.get(user_id) # Fetch user by ID

        if user and user.account_type == 'premium':
            user.account_type = 'basic'
            db.session.commit()
            print('Downgrade completed')
            return '', 204  # Return no content
        else:
            return '', 400  # Bad request if downgrade fails
    