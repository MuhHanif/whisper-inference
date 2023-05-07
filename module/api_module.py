# main api function using fast api
from fastapi import FastAPI, File, UploadFile, HTTPException
from queue import Queue
import os
import configparser
import json
import time
from .inference_module import *

with open("config_file.json") as f:
    config = json.load(f)

app = FastAPI()
# might be faster if using queue instead of storing as a file
# audio_queue = Queue()

# pin the model into memory
model = load_model(config["models"])


@app.get("/testing/{name}")
async def test_connection(name: str):
    return {"message": f"Hello, {name}!"}


@app.post("/upload/audio/")
async def upload_audio_file(file: UploadFile = File(...)) -> dict:
    """
    Endpoint that accepts an audio file and saves it to a specified folder.

    Args:
        file (UploadFile): The audio file to upload.

    Returns:
        dict: A JSON object containing the filename of the uploaded file.
    """
    # Specify the folder where the file will be saved
    upload_folder = "/content/uploads"

    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Check if the uploaded file is an audio file
    valid_extensions = [".mp3", ".wav", ".ogg"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Save the file to the specified folder
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    output = whisper_to_vtt(model, file_path, output_dir="/kaggle/working/test")

    # im disabling using queue for now
    # contents: bytes = await file.read()
    # audio_queue.put(contents)

    return output


@app.post("/upload/audio_test_upload/")
async def upload_audio_file_test_upload(file: UploadFile = File(...)) -> dict:
    """
    Endpoint that accepts an audio file and saves it to a specified folder.

    Args:
        file (UploadFile): The audio file to upload.

    Returns:
        dict: A JSON object containing the filename of the uploaded file.
    """
    # Specify the folder where the file will be saved
    upload_folder = "/content/uploads"

    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Check if the uploaded file is an audio file
    valid_extensions = [".mp3", ".wav", ".ogg"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Save the file to the specified folder
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # im disabling using queue for now
    # contents: bytes = await file.read()
    # audio_queue.put(contents)

    return {"message": "working as intended"}


@app.post("/upload/audio_test_preprocessing/")
async def upload_audio_file_test_preprocessing(file: UploadFile = File(...)) -> dict:
    """
    Endpoint that accepts an audio file and saves it to a specified folder.

    Args:
        file (UploadFile): The audio file to upload.

    Returns:
        dict: A JSON object containing the filename of the uploaded file.
    """
    # Specify the folder where the file will be saved
    upload_folder = "/content/uploads"

    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Check if the uploaded file is an audio file
    valid_extensions = [".mp3", ".wav", ".ogg"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Save the file to the specified folder
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    tic = time.time()
    audio = split_stereo_audio_ffmpeg(file_path)
    tac = time.time()
    # im disabling using queue for now
    # contents: bytes = await file.read()
    # audio_queue.put(contents)

    return {"message": f"processing audio took {tac-tic} second(s)"}
