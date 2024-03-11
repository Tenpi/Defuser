import shutil
import os
import platform

dirname = os.path.dirname(__file__)

name = "defuzers"
tmp_dir = "tmp"
build_dir = "build"
dist_dir = "dist"
app_dir = "server"
app = "server"
if platform.system() == "Windows":
    app = "server.exe"

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

if __name__ == "__main__":
    os.remove(os.path.join(dirname, "server.spec"))
    shutil.rmtree(os.path.join(dirname, tmp_dir))
    shutil.copytree(os.path.join(dirname, dist_dir), os.path.join(dirname, build_dir, name, dist_dir))
    shutil.copytree(os.path.join(dirname, "dialog"), os.path.join(dirname, build_dir, name, "dialog"))
    shutil.move(os.path.join(dirname, build_dir, app_dir), os.path.join(dirname, build_dir, name, app_dir))
    shutil.copy(os.path.join(dirname, "config.json"), os.path.join(dirname, build_dir, name, "config.json"))

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

    shutil.make_archive(name, "zip", os.path.join(dirname, build_dir), name)
    shutil.move(os.path.join(dirname, f"{name}.zip"), os.path.join(dirname, build_dir, f"{name}.zip"))