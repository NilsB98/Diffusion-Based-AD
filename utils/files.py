import json
import os
import dataclasses


def save_args(args, directory: str, filename: str):
    if not os.path.exists(f"{directory}"):
        os.makedirs(f"{directory}")
    config_file = open(f"{directory}/{filename}.json", "w+")
    content = dataclasses.asdict(args) if dataclasses.is_dataclass(args) else args
    config_file.write(json.dumps(content))
    config_file.close()
