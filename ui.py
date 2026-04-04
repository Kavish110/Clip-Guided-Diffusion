import gradio as gr
import glob
import os
import time

def get_latest_images(result_dir="results"):
    # Recursively find pngs
    images = glob.glob(f"{result_dir}/**/*.png", recursive=True)
    # Sort files by modification time
    images.sort(key=os.path.getmtime, reverse=True)
    # Get top 10 most recent
    paths = images[:10]
    return paths

def get_loss_plot(result_dir="results"):
    loss_plots = glob.glob(f"{result_dir}/**/training_loss.png", recursive=True)
    if loss_plots:
        loss_plots.sort(key=os.path.getmtime, reverse=True)
        return loss_plots[0]
    return None

def auto_refresh():
    return get_latest_images(), get_loss_plot()

with gr.Blocks(title="DiffusionCLIP Tiny ImageNet Tracker") as demo:
    gr.Markdown("# DiffusionCLIP Training Monitor")
    gr.Markdown("View the real-time generated image transitions and loss plots.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Training Loss")
            loss_image = gr.Image(label="Loss Plot", type="filepath")
        with gr.Column():
            gr.Markdown("## Image Transitions")
            gallery = gr.Gallery(label="Latest Generated Images", show_label=False, elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto")
            
    btn = gr.Button("Refresh Data")
    btn.click(fn=auto_refresh, inputs=[], outputs=[gallery, loss_image])
    
    # Auto-refresh mechanism on load
    demo.load(fn=auto_refresh, inputs=[], outputs=[gallery, loss_image])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
