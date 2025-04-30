
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import subprocess
from threading import Thread

sys.path.insert(0, os.getcwd())

import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

MAX_IMAGES = 150

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error("Please upload at least 2 images to train your model")
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"Only {MAX_IMAGES} or less images are allowed for training")

    updates.append(gr.update(visible=True))
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(uploaded_images)
        updates.append(gr.update(visible=visible))
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))
        corresponding_caption = False
        if image_value:
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible else None
        updates.append(gr.update(value=text_value, visible=visible))
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return f"\"{os.path.normpath(os.path.join(current_dir, p))}\""

def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(current_dir, p))

def start_training(train_script, train_config, sample_prompts):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    file_type = "sh" if sys.platform != "win32" else "bat"
    sh_filename = f"train.{file_type}"
    with open(sh_filename, "w", encoding="utf-8") as f:
        f.write(train_script)
    with open("dataset.toml", "w", encoding="utf-8") as f:
        f.write(train_config)
    with open("sample_prompts.txt", "w", encoding="utf-8") as f:
        f.write(sample_prompts)

    command = resolve_path_without_quotes(sh_filename)
    if sys.platform != "win32":
        command = f"bash {command}"

    process = subprocess.Popen(
        command,
        shell=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    for line in process.stdout:
        yield line

with gr.Blocks(title="FluxGym (LogsView-Free)", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown("## Terminal Output (Logs)")
    with gr.Row():
        output_log = gr.Textbox(label="Output", lines=20, interactive=False)

    def dummy_train():
        for i in range(5):
            yield f"Linha {i+1}: execução de teste..."

    btn = gr.Button("Executar exemplo")
    btn.click(fn=dummy_train, outputs=output_log)

if __name__ == "__main__":
    demo.launch()
