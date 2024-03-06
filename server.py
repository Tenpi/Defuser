import flask                                      
from flask_socketio import SocketIO, emit
from routes.functions import get_models_dir
import os

app = flask.Flask(__name__, static_url_path="", static_folder="dist")
app.secret_key = "klee"
socketio = SocketIO(app)

import routes.generate
import routes.interrogate
import routes.segmentate
import routes.controlnet
import routes.promptgen
import routes.info
import routes.train
import routes.misc

@app.route("/assets/<path:filename>")
def assets(filename):
    return flask.send_from_directory("assets", filename)

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return flask.send_from_directory("outputs", filename)

@app.route("/models/<path:filename>")
def models(filename):
    file_path = os.path.join(get_models_dir(), filename)
    print(file_path)
    return flask.send_file(file_path)

@app.route("/retrieve")
def retrieve():
    file_path = flask.request.args.get("path")
    if file_path:
        if "&" in file_path:
            file_path = file_path.split("&")[0]
        return flask.send_file(file_path)
    else:
        return "Path not provided", 400

@app.route("/")
def index():
    return flask.send_from_directory("dist", "index.html")

if __name__ == "__main__":
    host = "localhost"
    port = 8084
    print(f"* Running on http://{host}:{port}")
    socketio.run(app, host=host, port=port) 