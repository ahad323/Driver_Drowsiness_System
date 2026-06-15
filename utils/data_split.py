import os
import random
import shutil

base_path = "data/yawn"

# Step 1: create folders
for split in ["train", "val", "test"]:
    for category in ["yawn", "no yawn"]:   # <-- IMPORTANT: no yawn (no underscore)
        os.makedirs(os.path.join(base_path, split, category), exist_ok=True)

# Step 2: split function
def split(category):
    src = os.path.join(base_path, category)

    print("Reading from:", src)  # debug

    files = os.listdir(src)
    random.shuffle(files)

    total = len(files)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    for i, file in enumerate(files):
        src_file = os.path.join(src, file)

        if i < train_end:
            dst = os.path.join(base_path, "train", category, file)
        elif i < val_end:
            dst = os.path.join(base_path, "val", category, file)
        else:
            dst = os.path.join(base_path, "test", category, file)

        shutil.copy(src_file, dst)

# Step 3: run
split("yawn")
split("no yawn")

print("DONE ✅")