import shutil
import os
import platform
import json
import zipfile

dirname = os.path.dirname(__file__)

name = "defuzers"
tmp_dir = "tmp"
build_dir = "build"
dist_dir = "dist"
app_dir = "main"
app = "main"
if platform.system() == "Windows":
    app = "main.exe"

make_dirs = [
    "outputs",
    "models/animatediff",
    "models/controlnet/annotator",
    "models/controlnet/canny",
    "models/controlnet/depth",
    "models/controlnet/lineart",
    "models/controlnet/lineart anime",
    "models/controlnet/scribble",
    "models/controlnet/softedge",
    "models/detector",
    "models/diffusion",
    "models/hypernetworks",
    "models/interrogator",
    "models/ipadapter/models",
    "models/ipadapter/sdxl_models",
    "models/lora",
    "models/misc",
    "models/promptgen",
    "models/segmentator",
    "models/textual inversion",
    "models/upscaler",
    "models/vae"
]

def zip_file(input_dir, output_zip):
    """Zip up a directory and preserve symlinks and empty directories"""
    zip_out = zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED)
    rootLen = len(os.path.dirname(input_dir))
    def _ArchiveDirectory(parentDirectory):
        contents = os.listdir(parentDirectory)
        if not contents:
            archiveRoot = parentDirectory[rootLen:].replace("\\", "/").lstrip("/")
            zipInfo = zipfile.ZipInfo(archiveRoot+"/")
            zip_out.writestr(zipInfo, "")
        for item in contents:
            fullPath = os.path.join(parentDirectory, item)
            if os.path.isdir(fullPath) and not os.path.islink(fullPath):
                _ArchiveDirectory(fullPath)
            else:
                archiveRoot = fullPath[rootLen:].replace("\\", "/").lstrip("/")
                if os.path.islink(fullPath):
                    zipInfo = zipfile.ZipInfo(archiveRoot)
                    zipInfo.create_system = 3
                    zipInfo.external_attr = 2716663808
                    zip_out.writestr(zipInfo, os.readlink(fullPath))
                else:
                    zip_out.write(fullPath, archiveRoot, zipfile.ZIP_DEFLATED)
    _ArchiveDirectory(input_dir)
    zip_out.close()

if __name__ == "__main__":
    os.remove(os.path.join(dirname, "main.spec"))
    shutil.rmtree(os.path.join(dirname, tmp_dir))
    shutil.copytree(os.path.join(dirname, dist_dir), os.path.join(dirname, build_dir, name, dist_dir))
    shutil.copytree(os.path.join(dirname, "dialog"), os.path.join(dirname, build_dir, name, "dialog"))
    shutil.move(os.path.join(dirname, build_dir, app_dir), os.path.join(dirname, build_dir, name, app_dir))
    shutil.copy(os.path.join(dirname, "config.json"), os.path.join(dirname, build_dir, name, "config.json"))
    os.remove(os.path.join(dirname, build_dir, name, dist_dir, "assets/images/patch.png"))
    os.remove(os.path.join(dirname, build_dir, name, dist_dir, "assets/images/readme.png"))

    for dir in make_dirs:
        os.makedirs(os.path.join(dirname, build_dir, name, dir), exist_ok=True)

    os.rename(os.path.join(dirname, build_dir, name, app_dir), os.path.join(dirname, build_dir, name, "tmp"))
    shutil.move(os.path.join(dirname, build_dir, name, "tmp", app), os.path.join(dirname, build_dir, name, app))
    shutil.move(os.path.join(dirname, build_dir, name, "tmp", "_internal"), os.path.join(dirname, build_dir, name, "_internal"))
    shutil.rmtree(os.path.join(dirname, build_dir, name, "tmp"))

    index_path = os.path.join(dirname, build_dir, name, dist_dir, "index.html")
    with open(index_path, "r+") as index:
        data = index.read()
        data = data.replace("script.js", "dist/script.js")
        data = data.replace("styles.css", "dist/styles.css")
        index.seek(0)
        index.write(data)

    package_path = os.path.join(dirname, "package.json")
    config_path = os.path.join(dirname, build_dir, name, "config.json")
    version = ""
    repo_url = ""
    with open(package_path) as pkg:
        json_data = json.load(pkg)
        version = json_data["version"]
        repo_url = json_data["repository"]["url"]
    with open(config_path, "r+") as cfg:
        json_dict = {"version": version, "repository": {"url": repo_url}}
        json_data = json.load(cfg)
        for key, value in json_data.items():
            json_dict[key] = value
        cfg.seek(0)
        json.dump(json_dict, cfg, indent=4)

    zip_file(os.path.join(dirname, build_dir, name), os.path.join(dirname, build_dir, f"{name}.zip"))