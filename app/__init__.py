from flask import Flask

app = Flask(__name__)
print("entering init")
from app import views