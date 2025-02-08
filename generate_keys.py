import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ["Administrator", "Jeremy Omwenga"]
emails = ["admin1@gmail.com", "jeremyangwenyi.ja@gmail.com"]
usernames = ["admin", "jeremy1234"]
passwords = ["XXXX", "XXXXXX"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"

with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)