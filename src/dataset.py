import json
from collections import defaultdict
from image import Image
from metrics import Metrics
import pathlib
import os


class Alias:
    def __init__(self, class_id, class_name, alias, metrics):
        self.class_id = class_id
        self.class_name = class_name
        self.alias = ""
        self.combinations = None
        self.metrics: Metrics = metrics

    def parse(self, alias):
        # if alias has a . (dot), split the alias and set combinations to True
        if "." in alias:
            self.alias = alias
            self.combinations = alias.split(".")

    def validate(self):
        # check that all the alias characters are alphabetical
        if self.combinations:
            return all(
                all(char.isalpha() for char in comb) for comb in self.combinations
            )
        return all(char.isalpha() for char in self.alias)


class Dataset:
    def __init__(self, main_dir: str):
        self.path = pathlib.Path(main_dir)
        self.images = []
        self.aliases = []
        self.classnames = self.classes

    def add_image(self, image: Image):
        if isinstance(image, Image):
            self.images.append(image)
        else:
            raise ValueError("Image object is expected")

    def add_alias(self, alias: Alias):
        if isinstance(alias, Alias):
            self.aliases.append(alias)
        else:
            raise ValueError("Alias object is expected")

    def populate(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".jpg"):
                    img = Image(os.path.join(root, file))
                    self.add_image(img)

    @property
    def classes(self):
        cls_path = self.path / "classes.txt"
        with open(cls_path, "r") as f:
            return f.readlines()

    @property
    def history(self):
        # get all aliases from the dataset
        history = defaultdict(dict)
        for alias in self.aliases:
            history[alias.class_name][alias.alias] = alias.metrics
        return history

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
