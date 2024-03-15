import flask                                      
from flask_socketio import SocketIO, emit
from routes.functions import get_models_dir, get_outputs_dir
from engineio.async_drivers import threading
import os
import json
import logging
import signal
import requests

signal.signal(signal.SIGINT, signal.SIG_DFL)

dirname = os.path.dirname(__file__)
if "_internal" in dirname: dirname = os.path.join(dirname, "../")
if "Frameworks" in dirname: dirname = os.path.normpath(os.path.join(dirname, "../Resources"))

app = flask.Flask(__name__, static_url_path="", static_folder="dist")
app.secret_key = "klee"
socketio = SocketIO(app, async_mode="threading")
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
host = "localhost"
port = 8084

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
    file_path = os.path.normpath(os.path.join(dirname, "dist/assets", filename))
    return flask.send_file(file_path)

@app.route("/dist/<path:filename>")
def dist(filename):
    file_path = os.path.normpath(os.path.join(dirname, "dist", filename))
    return flask.send_file(file_path)

@app.route("/outputs/<path:filename>")
def outputs(filename):
    file_path = os.path.normpath(os.path.join(get_outputs_dir(), filename))
    return flask.send_file(file_path)

@app.route("/models/<path:filename>")
def models(filename):
    file_path = os.path.normpath(os.path.join(get_models_dir(), filename))
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
    file_path = os.path.normpath(os.path.join(dirname, "dist", "index.html"))
    return flask.send_file(file_path)

def load_config():
    global host
    global port
    config_path = os.path.normpath(os.path.join(dirname, "config.json"))
    if os.path.exists(config_path):
        with open(config_path) as config:
            data = json.load(config)
        host = data["host"]
        port = data["port"]

if __name__ == "__main__":
    load_config()
    try:
        requests.get(f"http://{host}:{port}/ping")
    except:
        print(f"* Running on http://{host}:{port}")
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)