import gradio as gr
import json
from module.inference_module import *
import os

whisper_config = "config_file.json"
json_config = "conf.json"

# Get the absolute path to the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_path = os.path.dirname(dir_path)
# Define the path to the config file relative to the script directory
config_path = os.path.join(dir_path, json_config)
whisper_config_path = os.path.join(dir_path, whisper_config)

# read config file for credentials and queue
with open(config_path, "r") as conf:
    config = json.load(conf)

# read config file for whisper models
with open(whisper_config_path, "r") as whisper_conf:
    whisper_config = json.load(whisper_conf)

# pin the model into memory
model = load_model(whisper_config["models"], device=whisper_config["cuda_device"])


def transcribe_audio(audio_file):
    file_name = audio_file.name

    output = whisper_to_vtt(
        model,
        file_name,
        # TODO! put this output dir in the config
        output_dir="/kaggle/working/test",
        channel_uid=[f"{'speaker1'}:", f"{'speaker2'}:"],
    )

    return output


# gradio ui input output
inputs = gr.inputs.File()
outputs = gr.outputs.Textbox()


# start interface and executing defined function
gr.Interface(
    fn=transcribe_audio,
    inputs=inputs,
    outputs=outputs,
    title="Whisper Testing Sequential",
    allow_flagging="never",
).launch(server_name="0.0.0.0", server_port=9789)
