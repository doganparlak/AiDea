from flask import Flask
from config import Config
from tables import db, TrainedModels, SubscriptionPlan, User
from routes import init_routes
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from flask_wtf.csrf import CSRFProtect
from datetime import datetime, timedelta

# Initialize CSRF protection
csrf = CSRFProtect()

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
        # Fetch all users whose subscriptions are expiring 
        expiring_users = User.query.filter(
            User.account_type != 'basic',
            User.subscription_end_date == datetime.utcnow() 
        ).all()

        for user in expiring_users:
            # Renew the subscription by extending the end date
            if user.account_type == 'monthly':
                user.subscription_end_date += timedelta(days=30)
            elif user.account_type == 'quarterly':
                user.subscription_end_date += timedelta(days=90)
            elif user.account_type == 'yearly':
                user.subscription_end_date += timedelta(days=365)

            # Process payment if applicable (placeholder for payment processing logic)
            # Example: process_payment(user)
            
            db.session.commit()  # Commit the changes

def downgrade_nonrenewed_subscriptions(): 
    with app.app_context():
        # Fetch all users with premium accounts and downgraded to basic accounts
        users = User.query.filter(User.account_type != 'basic').all()
        for user in users:
            # Check if the subscription end date is current day and no renewal is requested
            if user.subscription_end_date == datetime.utcnow() and user.renewal == False:  
                user.account_type = 'basic'  # Downgrade to basic
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
        scheduler.add_job(func=downgrade_nonrenewed_subscriptions, trigger="interval", days=1)

        # Job to renew subscriptions every day
        scheduler.add_job(func=renew_expiring_subscriptions, trigger="interval", days=1)

        scheduler.start()

        # Shut down the scheduler when exiting the app
        atexit.register(lambda: scheduler.shutdown())
    else:
        print("Scheduler is already running.")

# Initialize Flask app and Scheduler
app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_plans()  # Seed the subscription plans into the database
        create_scheduler(app)

    app.run(debug=True)
