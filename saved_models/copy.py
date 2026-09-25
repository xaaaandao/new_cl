import os
import pathlib
import shutil

for p in pathlib.Path(".").rglob("*loss.csv"):
    filename = "./" + p.parent.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    if not os.path.exists(filename):
        shutil.copy(p, filename)

for p in pathlib.Path(".").rglob("*results+*.csv"):
    filename = "./" + p.parent.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    if not os.path.exists(filename):
        shutil.copy(p, filename)

for p in pathlib.Path("./").rglob("features"):
    filename = "./" + p.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    if not os.path.exists(filename):
        shutil.copytree(p, filename)
