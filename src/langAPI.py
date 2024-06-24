import os
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

chat = ChatGroq(temperature=0, model="gemma-7b-it")

system = "Sei un assistente in uno studio legale e quando ti vengono sottoposti delle domande da parte dei clienti prepari le informazioni che serviranno ad un avvocato"


async def predict(question, history):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Scrivi normative e numeri delle sentenze della giurisprudenza italiana di riferimento riguardanti il contesto fornito dal cliente: {topic}",
            ),
        ]
    )
    chain = prompt | chat
    answer = await chain.ainvoke({"topic": question})
    return answer.content


gr.ChatInterface(predict).launch()


examples = [
    {
        "question": """
        Given the AveragePrecision of the past aliases:
        history:{
        "box": 0.3,
        "packaging": 0.4,
        "pallet": 0.5,
        "container": 0.2,
        }
        find a new alias to increase the performance of a Grounded Object Detector.""",
        "answer": """crate""",
    },
    {
        "question": """
        Given the AveragePrecision of the past aliases:
        history:{
        "box": 0.3,
        "packaging": 0.4,
        "crate": 0.5,
        }
        find a new alias to increase the performance of a Grounded Object Detector.""",
        "answer": """container""",
    },
]


example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

history = {
    "box": 0.3,
    "packaging": 0.4,
    "pallet": 0.5,
    "container": 0.2,
}

input = (
    "Given the AveragePrecision of the past aliases:\n"
    + "history:{"
    + str(history)
    + "}"
    + "find a new alias to increase the performance of a Grounded Object Detector."
)
print(prompt.format(input=input))
