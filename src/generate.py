import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import json
from collections import defaultdict

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set")
os.environ["GROQ_API_KEY"] = API_KEY

HISTORY_PATH = "/Users/francescotacinelli/Developer/automata/data/alias_history.json"

classes = ["bitter pack", "bottle pack", "box", "can pack", "crate", "keg"]
class_descriptions = {
    "bitter pack": "a pack of bitters, like crodino or tonic water",
    "bottle pack": "a pack of plastic bottles in a plastic wrap",
    "box": "a cardboard box",
    "can pack": "a pack of metal cans like beer or soda cans",
    "crate": "a plastic crate for bottles",
    "keg": "a metal keg like a beer keg or a gas canister",
}


class AliasGenerator:
    def __init__(self, path: str = HISTORY_PATH):
        self.path = os.getenv("HISTORY_PATH") or path
        self.history = defaultdict(tuple[float, float])
        self.chat = ChatGroq(temperature=0, model="gemma-7b-it")
        self.load()

    def load(self):
        with open(self.path, "r") as file:
            self.history = json.load(file)

    def update_history(self, alias: str, performance: tuple[float, float]):
        self.history[alias] = performance

    def should_continue(self, final_metrics: tuple[float, float]):
        # if recall is less than 0.5, continue
        if final_metrics[1] < 0.5:
            return True
        return False

    @property
    def best(self) -> tuple:
        best_precision = 0
        best_recall = 0
        best_pr_alias = ""
        best_rc_alias = ""
        for alias, (precision, recall) in self.history.items():
            if precision > best_precision:
                best_pr_alias = alias
                best_precision = precision
            if recall > best_recall:
                best_rc_alias = alias
                best_recall = recall
        return best_pr_alias, best_rc_alias

    def save(self, path: str):
        with open(path, "w") as file:
            json.dump(self.history, file)

    def improve(self, performance, class_id):
        return self.generate(performance, json.dumps(self.history), class_id)

    def generate(self, performance, history, class_id):
        """
        Args:
            input (str): The performance of the alias.
            history (list[tuple[str,str]]): The history of aliases and their performance.
                input = ""
                user_message = "performance of the alias"
                bot_message = "alias"
                history = [[user_message, bot_message]]
        Returns:
            new_alias (str): The best alias to increase the performance.
        """

        class_description = class_descriptions[classes[class_id]]
        system = f"We're trying to find the best aliases (to detect {class_description}) for a grounded object detector. Provide one alias at a time, then I will provide the precision and recall of the provided alias. With that in mind provide an alias to increase the performance."
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "performance: {performance}, history: {history}"),
            ]
        )
        chain = prompt | self.chat
        answer = chain.invoke({"performance": performance, "history": history})
        new_alias = str(answer.content)
        return new_alias
