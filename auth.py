# auth.py
from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create a user class
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Set up a simple user database (replace with a real database in production)
users = {'user@example.com': {'password': 'password'}}

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email]['password'] == password:
            user = User(email)
            login_user(user)
            return redirect(url_for('main.index'))
        else:
            return 'Invalid email or password'
    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth.route('/update', methods=['GET', 'POST'])
@login_required
def update():
    if request.method == 'POST':
        # Handle updating user information here
        pass
    return render_template('update.html')

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email not in users:
            users[email] = {'password': password}
            return redirect(url_for('auth.login'))
        else:
            return 'Email already exists'
    return render_template('register.html')
