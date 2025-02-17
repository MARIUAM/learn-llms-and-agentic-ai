import gradio as gr

def helloWorld(name):
    print(f"helloWorld called with {name}   ")
    return "Hello " + name + "!"

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

gr.Interface(fn=helloWorld, inputs="text", outputs="text", js=force_dark_mode).launch()