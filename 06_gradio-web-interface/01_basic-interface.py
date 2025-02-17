import gradio as gr

def helloWorld(name):
    print(f"helloWorld called with {name}   ")
    return "Hello " + name + "!"

gr.Interface(fn=helloWorld, inputs="text", outputs="text").launch(share=True)