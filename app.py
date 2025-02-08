import streamlit as st
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import pickle 

names = ["Administrator", "Jeremy Omwenga"]
emails = ["admin1@gmail.com", "jeremyangwenyi.ja@gmail.com"]
usernames = ["admin", "jeremy1234"]

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"

with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {"usernames":{}}

for un, mail, name, pw in zip(usernames, emails, names, hashed_passwords):
    user_dict = {"name":name, "mail":mail, "password":pw}
    credentials["usernames"].update({un:user_dict})


authenticator = stauth.Authenticate(credentials,"dashboard_module", "abcdef", cookie_expiry_days=1)

name, authentication_status, username = authenticator.login('sidebar', fields = {'Form name': 'Login','Email': 'Email'})

st.header("Welcome to Stock Forecast")





if authentication_status == False:
    st.error("Username/password is incorrect")

##if authentication_status == None:
    ##st.warning("Please enter your username and password")

if authentication_status:

	

	authenticator.logout("Log Out", "main")

	st.title(f"Welcome, {name}")

	about_page = st.Page(
	page = "pages/navigation.py",
	title = "About Stock Price Prediction",
	icon = ":material/account_circle:",
	default = True,
	)

	
