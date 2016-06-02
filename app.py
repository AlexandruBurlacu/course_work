from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/logout")
def logout():
    return render_template("logout.html")
    
@app.route("/login")
def login():
    return render_template("login.html")

if __name__ == "__main__":
    app.run("0.0.0.0", 8080, debug = True)