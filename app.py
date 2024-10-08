from flask import Flask
from config import Config
from tables import db, TrainedModels, SubscriptionPlan, User
from routes import init_routes, get_plan_amount, read_settings
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import stripe
from flask_wtf.csrf import CSRFProtect
from datetime import datetime, timedelta
import iyzipay
import json

# Initialize CSRF protection
csrf = CSRFProtect()
# Load the settings file
settings = read_settings('settings.yaml')
iyzico_api_key = settings['IYZICO-KEYS']['api_key']
iyzico_secret_key = settings['IYZICO-KEYS']['api_key']
iyzico_base_url = settings['IYZICO-KEYS']['base_url']
options = {
    'api_key': iyzico_api_key,
    'secret_key': iyzico_secret_key,
    'base_url': iyzico_base_url
}

def create_app():
    app = Flask(__name__)
    csrf.init_app(app)  # Enable CSRF protection for the app
    app.config.from_object(Config)
    
    db.init_app(app)
    # Initialize routes
    init_routes(app)
    
    return app

def renew_expiring_subscriptions():
    with app.app_context():
        # Get the current time rounded to the nearest minute
        today = datetime.utcnow().replace(second=0, microsecond=0)
        # Fetch all users whose subscriptions are expiring 
        users = User.query.filter(
            User.account_type != 'basic',
            User.subscription_end_date <= today,
            User.renewal == True
        ).all()
        for user in users:
            # Get amount to be deducted
            amount = get_plan_amount(user.account_type)
            if not user.card_user_key or not user.card_token:
                print(f"No payment token found for user {user.email}, skipping renewal.")
                continue

            # Iyzico payment request for auto-renewal
            request_data = {
                'locale': 'tr',  # Turkish language
                'conversationId': str(user.id),
                'price': str(amount),
                'paidPrice': str(amount),
                'currency': 'TRY',
                'basketId': str(user.id),
                'paymentCard': {
                    'cardToken': user.card_token,  # Use the stored card token
                    'cardUserKey': user.card_user_key  # Use the stored cardUserKey
                },
                'buyer': {
                    'id': str(user.id),
                    'email': user.email,
                },
                'basketItems': [
                    {
                        'id': '1',
                        'name': user.account_type,
                        'category1': 'Subscription',
                        'itemType': 'VIRTUAL',
                        'price': str(amount)
                    }
                ]
            }

            # Make the request to Iyzico
            payment_result = iyzipay.Payment().create(request_data, options)
            result_json = payment_result.read().decode('utf-8')
            result = json.loads(result_json)

            if result['status'] == 'success':
                print(f"Subscription renewed successfully for {user.email}")

                # Extend the subscription end date
                if user.account_type == 'monthly':
                    user.subscription_end_date += timedelta(days=30)
                elif user.account_type == 'quarterly':
                    user.subscription_end_date += timedelta(days=90)
                elif user.account_type == 'yearly':
                    user.subscription_end_date += timedelta(days=365)
                elif user.account_type == 'minutely':
                    user.subscription_end_date += timedelta(minutes=2)

                # Commit the renewal in the database
                db.session.commit()

            else:
                # Downgrade to basic if payment is not successful
                user.account_type = 'basic'  # Downgrade to basic
                user.subscription_end_date = None
                db.session.commit()  # Commit the changes
                print(f"Failed to renew subscription for {user.email}: {result.get('errorMessage')}")
        


def downgrade_nonrenewed_subscriptions(): 
    with app.app_context():
        # Get the current time rounded to the nearest minute
        today = datetime.utcnow().replace(second=0, microsecond=0)
        # Fetch all users with premium accounts and downgraded to basic accounts
        users = User.query.filter(
                User.account_type != 'basic',
                User.subscription_end_date <= today,
                User.renewal == False
        ).all()
        for user in users:
            user.account_type = 'basic'  # Downgrade to basic
            user.subscription_end_date = None
            db.session.commit()  # Commit the changes

# Function to seed subscription plans into the database
def seed_plans():
    plans = [
        SubscriptionPlan(name='monthly', price=19.99, duration_in_days=30),
        SubscriptionPlan(name='quarterly', price=39.99, duration_in_days=90),
        SubscriptionPlan(name='yearly', price=99.99, duration_in_days=365)
    ]
    # Ensure the plans are not duplicated
    for plan in plans:
        existing_plan = SubscriptionPlan.query.filter_by(name=plan.name).first()
        if not existing_plan:
            db.session.add(plan)
    db.session.commit()

# Wrapper function to call the TrainedModels.delete_all_models method within the app context
def delete_all_models_wrapper():
    with app.app_context():  # Ensure the function is running within the app context
        TrainedModels.delete_all_models()

# Function to create and start the scheduler
def create_scheduler(app):
    scheduler = BackgroundScheduler()
    if not scheduler.running:
        # Pass the wrapper function to the scheduler, delete all models after 2 days
        scheduler.add_job(func=delete_all_models_wrapper, trigger="interval", days=2)

        # Add job to downgrade subscriptions daily
        scheduler.add_job(func=downgrade_nonrenewed_subscriptions, trigger="interval", minutes=1)

        # Job to renew subscriptions every day
        scheduler.add_job(func=renew_expiring_subscriptions, trigger="interval", minutes=1)

        scheduler.start()

        # Shut down the scheduler when exiting the app
        atexit.register(lambda: scheduler.shutdown())
    else:
        print("Scheduler is already running.")

def print_all_users():
    users = User.query.all()  # Fetch all users from the database
    if not users:
        print("No users found in the database.")
    else:
        print("Users in the database:")
        for user in users:
            print(user)  # This will call the __repr__ method of the User class

# Initialize Flask app and Scheduler
app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_plans()  # Seed the subscription plans into the database
        create_scheduler(app)
        print_all_users()  # Print all users in the database

    app.run(debug=True)
