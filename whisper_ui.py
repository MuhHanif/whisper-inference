import gradio as gr


def transcribe_audio(audio_file):
    print("+++++++++[testing]+++++++++")
    text = "testing"
    return text


audio_input = gr.inputs.Audio(label="Upload Audio File")
output_text = gr.outputs.Textbox(label="Transcription")

gr.Interface(
    fn=transcribe_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription",
    description="Transcribe an uploaded audio file to text.",
).launch()
