import pathlib
import shutil

for p in pathlib.Path(".").rglob("*loss.csv"):
    filename = "./saved_models/" + p.parent.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    shutil.copy(p, filename)

for p in pathlib.Path(".").rglob("*results.csv"):
    filename = "./saved_models/" + p.parent.parent.name + "_" + p.name
    print(f"src: {p}")
    print(f"dst: {filename}")
    shutil.copy(p, filename)
