import os
import pathlib
import shutil

for p in pathlib.Path(".").rglob("*loss.csv"):
    filename = "./saved_models/" + p.parent.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    shutil.copy(p, filename)

for p in pathlib.Path(".").rglob("*results+*.csv"):
    filename = "./saved_models/" + p.parent.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    shutil.copy(p, filename)

for p in pathlib.Path(".").rglob("*.npz"):
    dst = "./saved_models/" + p.parent.parent.parent.name + "_npz"
    print(f"src: {p}")
    os.makedirs(dst, exist_ok=True)
    filename = os.path.join(dst, p.name)
    print(f"dst: {filename}")
    shutil.copy(p, filename)

