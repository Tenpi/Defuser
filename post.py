import shutil
import os
import platform
import json
import zipfile
import argparse
import pathlib

dirname = os.path.dirname(__file__)

name = "img-diffuse"
name_cap = "Img Diffuse"
tmp_dir = "tmp"
build_dir = "build"
app_bundle_dir = "app"
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

def build_zip():
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

def build_app():
    os.remove(os.path.join(dirname, "Img Diffuse.spec"))
    shutil.rmtree(os.path.join(dirname, tmp_dir))
    internal_resources_raw = os.listdir(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources"))
    internal_resources = list(map(lambda x: os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources", x), internal_resources_raw))
    internal_resources_dup = list(map(lambda x: os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks", x), internal_resources_raw))
    for i, internal_resource in enumerate(internal_resources):
        if not os.path.islink(internal_resource):
            if ".icns" in internal_resource: continue
            if os.path.isdir(internal_resource):
                tmp = os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks/{pathlib.Path(internal_resource).stem}_tmp")
                dir = os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks/{pathlib.Path(internal_resource).stem}")
                os.makedirs(tmp, exist_ok=True)
                shutil.copytree(internal_resource, tmp, symlinks=False, dirs_exist_ok=True)
                if os.path.islink(internal_resources_dup[i]):
                    os.remove(internal_resources_dup[i])
                else:
                    shutil.rmtree(internal_resources_dup[i])
                os.rename(tmp, dir)
            else:
                os.remove(internal_resources_dup[i])
                shutil.copy(internal_resource, os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks"))
    shutil.rmtree(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources"))
    os.makedirs(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources"), exist_ok=True)
    shutil.copy(os.path.join(dirname, f"assets/icons/{name_cap}.icns"), os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources/{name_cap}.icns"))

    copy_to_resources = ["dialog", "dist", "models", "outputs", "config.json"]
    for item in copy_to_resources:
        item_path = os.path.join(dirname, build_dir, name, item)
        if os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources/{item}"), symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy(item_path, os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Resources/{item}"))

    shutil.copy(os.path.join(dirname, build_dir, name, app), os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/MacOS/{app}"))
    
    os.rename(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks"), os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks2"))
    shutil.copytree(os.path.join(dirname, build_dir, name, "_internal"), os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks"), symlinks=True, dirs_exist_ok=True)
    #os.remove(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks/base_library.zip"))
    #shutil.copyfile(os.path.join(dirname, build_dir, name, "_internal", "base_library.zip"), os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks/base_library.zip"), follow_symlinks=True)

    internals = os.listdir(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks2"))
    internals = map(lambda x: os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks", x), internals)
    for internal in internals:
        if os.path.exists(internal):
            if os.path.isdir(internal):
                shutil.rmtree(internal)
            else:
                os.remove(internal)
    shutil.copytree(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks2"), os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks"), symlinks=True, dirs_exist_ok=True)
    shutil.rmtree(os.path.join(dirname, app_bundle_dir, f"{name_cap}.app/Contents/Frameworks2"))
    shutil.rmtree(os.path.join(dirname, app_bundle_dir, name_cap))

def build_exe():
    os.remove(os.path.join(dirname, "Img Diffuse.spec"))
    shutil.rmtree(os.path.join(dirname, tmp_dir))

    copy_to_resources = ["dialog", "dist", "models", "outputs", "config.json", "main.exe"]
    for item in copy_to_resources:
        item_path = os.path.join(dirname, build_dir, name, item)
        if os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(dirname, app_bundle_dir, f"{name_cap}/{item}"), symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy(item_path, os.path.join(dirname, app_bundle_dir, f"{name_cap}/{item}"))

    internals = os.listdir(os.path.join(dirname, build_dir, name, "_internal"))
    internals = map(lambda x: os.path.join(dirname, build_dir, name, "_internal", x), internals)
    for internal in internals:
        base = os.path.basename(internal)
        if os.path.isdir(internal):
            shutil.copytree(internal, os.path.join(dirname, app_bundle_dir, f"{name_cap}/_internal/{base}"), symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy(internal, os.path.join(dirname, app_bundle_dir, f"{name_cap}/_internal/{base}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Post Script")
    parser.add_argument("-a", "--app", action="store_true")
    parser.add_argument("-e", "--exe", action="store_true")
    args = parser.parse_args()

    if args.app:
        build_app()
    elif args.exe:
        build_exe()
    else:
        build_zip()