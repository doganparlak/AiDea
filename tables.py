from flask_sqlalchemy import SQLAlchemy
import pickle


db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)  # 'basic' or 'premium'
    symbols = db.relationship('Symbol', backref='user', lazy=True)

    def __repr__(self):
        return f'<User: {self.email} - Account Type: {self.account_type}>'
    

class Symbol(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False, unique=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<Symbol: {self.name}>'
    

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

class TemporaryPassword(db.Model):
    email = db.Column(db.String(120), primary_key=True, nullable=False)
    temp_password = db.Column(db.String(255), nullable=False)