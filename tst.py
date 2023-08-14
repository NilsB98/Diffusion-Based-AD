# imports
import os

SAVE_NAME = "checkpoints/hazelnut"
exists = os.path.exists(SAVE_NAME)
if not exists:
    os.makedirs(SAVE_NAME)
print(exists)