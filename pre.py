import os
import json

dirname = os.path.dirname(__file__)
if "_internal" in dirname: dirname = os.path.join(dirname, "../")
if "Frameworks" in dirname: dirname = os.path.normpath(os.path.join(dirname, "../Resources"))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def load_config():
    config_path = os.path.normpath(os.path.join(dirname, "config.json"))
    if os.path.exists(config_path):
        with open(config_path) as config:
            data = json.load(config)

        if data["unlimitedMemory"]:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

if __name__ == "__main__":
    load_config()