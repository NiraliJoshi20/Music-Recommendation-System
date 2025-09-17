import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os
from datetime import datetime
import re

# File to store user credentials (for demo purposes)
USERS_FILE = "users.json"

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# Function to load users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save users
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Function to validate login
def validate_login(username, password):
    users = load_users()
    return username in users and users[username]['password'] == password

# Function to validate email format
def is_valid_email(email):
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

# Function to check password strength
def check_password_strength(password):
    score = 0
    feedback = []

    # Check length
    if len(password) >= 8:
        score += 2
    elif len(password) >= 6:
        score += 1
    else:
        feedback.append("Password is too short (minimum 6 characters).")

    # Check for uppercase letters
    if any(c.isupper() for c in password):
        score += 1
    else:
        feedback.append("Add uppercase letters for stronger password.")

    # Check for lowercase letters
    if any(c.islower() for c in password):
        score += 1
    else:
        feedback.append("Add lowercase letters for stronger password.")

    # Check for digits
    if any(c.isdigit() for c in password):
        score += 1
    else:
        feedback.append("Add numbers for stronger password.")

    # Check for special characters
    if any(not c.isalnum() for c in password):
        score += 1
    else:
        feedback.append("Add special characters (e.g., !@#$%) for stronger password.")

    # Determine strength level
    if score >= 5:
        strength = "Strong"
        color = "green"
    elif score >= 3:
        strength = "Medium"
        color = "orange"
    else:
        strength = "Weak"
        color = "red"

    return strength, color, feedback

# Function to register user
def register_user(username, password, email):
    users = load_users()
    if username in users:
        return False, "Username already exists!"
    users[username] = {
        'password': password,  # In production, hash passwords!
        'email': email,
        'created_at': datetime.now().isoformat()
    }
    save_users(users)
    return True, "Registration successful!"

# Main app
def main_app():
    # Load model and scaler
    try:
        kmeans = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return

    # Load dataset
    try:
        songs_df = pd.read_csv("clustered_df.csv")
    except Exception as e:
        st.error(f"Error loading clustered_df.csv: {e}")
        return

    # Page title
    st.title("🎧 Mood-Based Music Recommendation System")
    st.subheader(f"Welcome, {st.session_state.username}! Adjust your current mood below:")

    # Sliders for 7 features used during training
    valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.3)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)

    # Recommend songs
    if st.button("🎶 Recommend Songs"):
        try:
            input_data = np.array([[valence, danceability, tempo, acousticness, liveness, speechiness, instrumentalness]])
            scaled_input = scaler.transform(input_data)

            cluster = kmeans.predict(scaled_input)[0]

            # Debug info (optional)
            st.write(" Predicted Cluster:", cluster)

            if 'Cluster' not in songs_df.columns:
                st.error(" 'Cluster' column not found in dataset!")
                return

            recommended_songs = songs_df[songs_df['Cluster'] == cluster]

            st.success(f"Songs from Mood Cluster: {cluster}")

            if recommended_songs.empty:
                st.warning("No songs found for this cluster.")
            else:
                # Show up to 10 songs
                for _, row in recommended_songs.sample(n=min(10, len(recommended_songs))).iterrows():
                    st.write(f" **{row['name']}** by *{row['artists']}*")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = 'login'
        st.rerun()

# Login page
def login_page():
    st.title("🎧 Mood-Based Music Recommendation System")
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if validate_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = 'main'
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.write("Don't have an account?", unsafe_allow_html=True)
    if st.button("Go to Signup"):
        st.session_state.page = 'signup'
        st.rerun()

# Signup page
def signup_page():
    st.title("Sign Up")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password", key="password_input")
    confirm_password = st.text_input("Confirm Password", type="password")

    # Display password strength
    if password:
        strength, color, feedback = check_password_strength(password)
        st.markdown(f"**Password Strength**: <span style='color:{color}'>{strength}</span>", unsafe_allow_html=True)
        if feedback:
            st.write("Password Feedback:")
            for comment in feedback:
                st.write(f"- {comment}")

    if st.button("Sign Up"):
        if not is_valid_email(email):
            st.error("Invalid email format! Please use a valid email address (e.g., user@example.com).")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        elif not username or not email or not password:
            st.error("All fields are required!")
        else:
            success, message = register_user(username, password, email)
            if success:
                st.success(message)
                st.session_state.page = 'login'
                st.rerun()
            else:
                st.error(message)
    st.write("Already have an account?", unsafe_allow_html=True)
    if st.button("Go to Login"):
        st.session_state.page = 'login'
        st.rerun()

# Page navigation
if st.session_state.logged_in:
    main_app()
elif st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'signup':
    signup_page()