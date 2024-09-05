from flask import Flask
from config import Config
from tables import db, TrainedModels
from routes import init_routes
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    db.init_app(app)
    # Initialize routes
    init_routes(app)
    
    return app

# Wrapper function to call the TrainedModels.delete_all_models method within the app context
def delete_all_models_wrapper():
    with app.app_context():  # Ensure the function is running within the app context
        TrainedModels.delete_all_models()

# Function to create and start the scheduler
def create_scheduler(app):
    scheduler = BackgroundScheduler()
    # Pass the wrapper function to the scheduler
    scheduler.add_job(func=delete_all_models_wrapper, trigger="interval", days=2)
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

# Initialize Flask app and Scheduler
app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print(TrainedModels.query.all())
        create_scheduler(app)

    app.run(debug=True)
