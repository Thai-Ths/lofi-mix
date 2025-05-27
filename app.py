import gradio as gr
from lofi_mix import create_gradio_interface

def main():
    interface = create_gradio_interface()
    interface.launch()

if __name__ == "__main__":
    main()

