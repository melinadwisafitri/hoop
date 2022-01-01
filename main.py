
from flask import Flask, render_template, Response

app = Flask(__name__, template_folder='template')


# how to call in route main
@app.route('/')
def anyname():
    """Video streaming home page."""
    return "<h1>Welcome to Geeks for Geeks</h1>"


if __name__ == "__main__":
    app.run(debug=True)