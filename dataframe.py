import gradio as gr

# Modified sample DataFrame data with a list of tuples for each row
sample_dataframe = {
    "data": [
        ("bitter pack", "bottle pack"),
        ("bottle pack", "bitter pack"),
        ("box", "bottle pack"),
        ("can pack", "bottle pack"),
        ("keg", "bottle pack"),
    ],
    "headers": ["before", "after"],
}

# Use the modified data structure directly without an extra dictionary
with gr.Blocks() as demo:
    comparison = gr.DataFrame(sample_dataframe)

if __name__ == "__main__":
    demo.launch()
