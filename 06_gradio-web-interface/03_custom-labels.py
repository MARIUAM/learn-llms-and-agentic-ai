import gradio as gr

def helloWorld(name):
    print(f"helloWorld called with {name}   ")
    return "Hello " + name + "!"


view = gr.Interface(
    fn=helloWorld,
    inputs=[gr.Textbox(label="Enter your name", lines=6)],
    outputs=[gr.Textbox(label="Response", lines=8)]
)
view.launch()