import json
import pathlib as path
from pathlib import Path
from dotenv import dotenv_values


def get_path():
    config = dotenv_values(
        ".env"
    )  # config = {"USER": "foo", "EMAIL": "foo@example.org"}
    fallback = path.Path(path.Path(__file__).parent / "data/dataset.json")
    try:
        dataset = config.get("AUTODATASET") or ""
        dataset = path.Path(dataset)
    except KeyError or FileNotFoundError:
        dataset = fallback

    if dataset.exists():
        return dataset
    else:
        print(f"No dataset file found, creating one at {dataset}")
        with open(dataset, "w") as file:
            json.dump({}, file)
    return dataset


class Dataset:
    def __init__(self, main_dir: Path = get_path()):
        self.path = str(main_dir.resolve())
        self.classes: dict[str, int] = {
            "bitters": 0,
            "bottles": 1,
            "corrugated": 2,
            "cans": 3,
            "keg": 4,
            "crate": 5,
        }
        self.captions: dict[int, list[str]] = {
            0: ["crown cork bottles"],
            1: [
                "plastic cap bottles",
            ],
            2: ["cardboard box"],
            3: ["cans pack"],
            4: [
                "metal keg or gas canister",
            ],
        }
        self.categories = {
            "crown corks": 0,
            "plastic caps": 1,
            "cardboard box": 2,
            "tin cans": 3,
            "metal keg or gas canister": 4,
        }
        self.prompt = f"which of these categories: {self.categories.keys()} better describes the object in the image?"

    def load(self):
        with open(self.path, "r") as f:
            data = json.load(f)
            self.__dict__.update(data)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def legend(self):
        return list(self.classes.keys())


if __name__ == "__main__":
    dataset = Dataset()
    dataset.load()
    print("classes:", dataset.classes, "\n")
    print("captions:", dataset.captions, "\n")
    print("categories:", dataset.categories, "\n")
    print("prompt:", dataset.prompt, "\n")
    dataset.save()
