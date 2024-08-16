import os

class Config:
    SECRET_KEY = 'Doggan98-ddd'
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(os.path.abspath(os.path.dirname(__file__)), "instance", "aidea.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False