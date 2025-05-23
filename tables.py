from flask_sqlalchemy import SQLAlchemy
import pickle


db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)  # 'basic', 'monthly', 'quarterly', 'yearly'
    renewal = db.Column(db.Boolean, default=False)  # Flag to indicate if the subscription should auto-renew
    symbols = db.relationship('Symbol', backref='user', lazy=True)
    subscription_end_date = db.Column(db.DateTime, nullable=True)  # To manage billing cycle
    card_user_key = db.Column(db.String(255), nullable=True)  # Iyzico cardUserKey for future payments
    card_token = db.Column(db.String(255), nullable=True)  # Iyzico cardToken for future payments

    def __repr__(self):
            return (f'<User: {self.email} \n'
                    f'Account Type: {self.account_type} \n'
                    f'Subscription End Date: {self.subscription_end_date} \n'
                    f'Renewal: {self.renewal} \n'
                    f'Card User Key: {self.card_user_key} \n'
                    f'Card Token: {self.card_token}>')
            
    @classmethod
    def delete_all_users():
        try:
            # Delete all entries from the Users table
            Symbol.query.delete()
            db.session.commit()
            print("All users are deleted successfully.")
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {str(e)}")
    
class SubscriptionPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)  # Plan names like 'monthly', 'quarterly', 'yearly'
    price = db.Column(db.Float, nullable=False)      # Plan price
    duration_in_days = db.Column(db.Integer, nullable=False)  # Duration of plan in days

    def __repr__(self):
        return f'<SubscriptionPlan: {self.name}, Price: {self.price}>'
    
class Symbol(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False, unique=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<Symbol: {self.name}>'
    
    @classmethod
    def delete_all_symbols():
        try:
            # Delete all entries from the Symbols table
            Symbol.query.delete()
            db.session.commit()
            print("All symbols are deleted successfully.")
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {str(e)}")
    

class TrainedModels(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String(20), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    start_date = db.Column(db.String(50), nullable=False)
    end_date = db.Column(db.String(50), nullable=False)  # Nullable if model is ongoing
    model_obj = db.Column(db.Text)  # Serialized model data or file path

    def __repr__(self):
        return f'<Model: {self.model_type} - Symbol: {self.symbol} - Start Date: {self.start_date} - End Date: {self.end_date}>'

    def save_trained_model(self, model_obj):
        """
        Save the trained model to the database.
        """
        self.model_obj = pickle.dumps(model_obj)
        db.session.add(self)
        db.session.commit()
    
    @classmethod
    def load_trained_model(self, model_type, start_date, end_date, symbol):
        model_entry = TrainedModels.query.filter_by(
                        model_type=model_type,
                        start_date=start_date,
                        end_date=end_date,
                        symbol=symbol
                    ).first()
        if model_entry:
            return pickle.loads(model_entry.model_obj)
        else:
            return None
    
    @classmethod
    def delete_all_models(self):
        try:
            # Delete all entries from the TrainedModels table
            TrainedModels.query.delete()
            db.session.commit()
            print("All models are deleted successfully.")
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {str(e)}")


class TemporaryPassword(db.Model):
    email = db.Column(db.String(120), primary_key=True, nullable=False)
    temp_password = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<User: {self.email} - Temporary Password: {self.temp_password}>'
    
    @classmethod
    def delete_all_temporary_passwords():
        try:
            # Delete all entries from the TemporaryPassword table
            TemporaryPassword.query.delete()
            db.session.commit()
            print("All temporary passwords are deleted successfully.")
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {str(e)}")
