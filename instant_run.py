# main api function using fast api
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from queue import Queue
import os
import configparser
import json
import time
import module
from module.inference_module import *
import uuid
from threading import Thread
import boto3
from botocore.client import Config
from pydantic import BaseModel
import urllib
import argparse
import uvicorn
import whisper

app = FastAPI()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse JSON configuration into a dictionary")
    
    # Add arguments for each field in the JSON
    parser.add_argument("--models", type=str, required=True, help="Specify the model name (e.g., 'medium')")
    parser.add_argument("--cuda_device", type=str, required=True, help="Specify the CUDA device (e.g., 'cuda:0')")
    parser.add_argument("--maximum_queue", type=int, required=True, help="Specify the maximum queue size")
    parser.add_argument("--output_vtt_dir", type=str, required=True, help="Specify the output VTT directory")
    parser.add_argument("--audio_folder", type=str, required=True, help="Specify the audio folder directory")
    parser.add_argument("--ephemeral_data_duration", type=int, required=True, help="Specify the ephemeral data duration")
    parser.add_argument("--ephemeral_check_period", type=int, required=True, help="Specify the ephemeral check period")
    parser.add_argument("--port", type=int, required=True, help="port number ie: 8888")
    parser.add_argument("--host", type=str, required=True, help="host ip ie: 0.0.0.0")

    args = parser.parse_args()
    
    # Create a dictionary from the parsed arguments
    config = {
        "models": args.models,
        "cuda_device": args.cuda_device,
        "maximum_queue": args.maximum_queue,
        "output_vtt_dir": args.output_vtt_dir,
        "audio_folder": args.audio_folder,
        "ephemeral_data_duration": args.ephemeral_data_duration,
        "ephemeral_check_period": args.ephemeral_check_period,
        "port": args.port,
        "host": args.host
    }

    return config


def load_model(model_name: str, device: str = "cuda:0"):
    """
    Loads a Whisper model to GPU

    BUG! when defining cuda device. whisper always tried to load to cuda:0!
    probably problem with load_model method

    Args:
        model_name (str): The name of the model to load.
        device (str): CUDA device to run on.
    Returns:
        whisper.model: The loaded Whisper model.
    """
    # Load the Whisper model
    cuda_device = torch.device(device)
    tic = time.time()
    model = module.load_model(model_name, device=cuda_device)
    toc = time.time()

    print(f"Model loaded, took {round(toc-tic, 2)} second(s)")

    return model

class upload_audio(BaseModel):
    """
    data class for upload_audio_to_queue_process
    """

    queue_id: str
    status: str


class queue_info(BaseModel):
    """
    data class for queue_length
    """

    queue_length: int
    max_queue_length: int


class queue_packet(BaseModel):
    status: int


def purge_data_periodially():
    """
    purge data from local disk and dict after x minutes (epehemeral data) when transcribtion either succeed or failed
    only check for every 1 minute

    # dictionary changed size during iteration
    """
    while True:
        try:
            for key, value in audio_file.items():
                # wow black formatter made a mess
                # basically it check if transcribtion status is "transcribed"
                # and it's longer than defined ephemeral duration it will delete
                # the file to freed up space in the worker machine.
                if (
                    value["status"] == "transcribed"
                    and time.time() - value["timestamp"]
                    > config["ephemeral_data_duration"]
                ):
                    # deleting audio file
                    os.remove(value["file"])

                    # delete dict
                    del audio_file[key]

                # delete failed transcribtion after x seconds
                if (
                    value["status"] == "failed to transcribe audio"
                    and time.time() - value["timestamp"]
                    > config["ephemeral_data_duration"]
                ):
                    # deleting audio file
                    os.remove(value["file"])
                    # delete dict
                    del audio_file[key]

        except Exception as purging_error:
            print(purging_error)

        # sleep periodically to not hog resources
        time.sleep(config["ephemeral_check_period"])


# wrap process and run this concurrently indefinitely forever and ever :P
def process_audio():
    # periodically check if there's a task
    while True:
        # this will block until there's another queue
        uuid = internal_queue.get()
        print("thread is running")
        # retrieve audio file path that has been downloaded
        file_path = audio_file[uuid]["file"]
        # switch status to processed to indicate if this file is currently being processed
        audio_file[uuid]["status"] = "processing"
        # put try catch block here so the thread wont fail
        try:
            # raise ValueError("manually raised failed transcribtion")
            # this function call return transcribtion as dict and write it to disk
            
            # use local GPU
            output = whisper_to_vtt(
                model=model,
                audio_file_path=file_path,
                output_dir=config["output_vtt_dir"],
                save_output_as_file=False,
            )  

            # store transcribed output
            audio_file[uuid]["status"] = "transcribed"
            audio_file[uuid]["output_as_text"] = output
        except Exception as error_message:
            output = error_message
            print(output)
            # store error message reason
            audio_file[uuid]["output_as_text"] = output
            audio_file[uuid]["status"] = "failed to transcribe audio"
        print()


@app.get("/")
async def test_connection():
    return {"message": f"whisper API v0.1, go to /docs/ for documentation"}


@app.post("/upload_to_queue_process/")
async def upload_audio_to_queue_process(file: UploadFile = File(...)) -> upload_audio:
    """
    Endpoint that accepts an audio file and push it to internal queue to be processed

    will return queue id that can be used on `/transcribtion_status/` endpoint to retrieve transcription

    Arg:
        file (UploadFile): Audio file to upload `.mp3`, `.ogg`, `.wav`.

    example output:
    `    {
        "queue_id": "68f541ba-db94-49f7-837d-502fba269a1a",
        "status": "stored"
        }
    `
    """

    # tell client if queue is full
    if internal_queue.full():
        raise HTTPException(
            status_code=503,
            detail=f"worker queue is full! there's {config['maximum_queue']} audio in the queue",
        )

    # Specify the folder where the file will be saved
    upload_folder = config["audio_folder"]

    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Check if the uploaded file is an audio file
    valid_extensions = [".mp3", ".wav", ".ogg"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="File must be an audio file")

    uuid_name = str(uuid.uuid4())

    # Save the file to the specified folder
    file_path = os.path.join(upload_folder, f"{uuid_name}{file_extension}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # put process in queue
    internal_queue.put(uuid_name)

    # store necessary data to be processed
    audio_file[uuid_name] = {}
    audio_file[uuid_name]["queue_id"] = uuid_name
    audio_file[uuid_name]["file"] = file_path
    # status can be (stored or processing)
    audio_file[uuid_name]["status"] = "stored"
    # time is required for other function to periodicaly check if
    # this task is being processed
    audio_file[uuid_name]["timestamp"] = time.time()

    # return {"queue_id": uuid_name, "status": audio_file[uuid_name]["status"]}
    return upload_audio(queue_id=uuid_name, status=audio_file[uuid_name]["status"])


@app.get("/transcription_status/{uuid_name}")
async def transcription_status(uuid_name: str):
    """"""
    try:
        # TODO! put this in struct
        return audio_file[uuid_name]
    except KeyError:
        raise HTTPException(status_code=404, detail="UUID not found")


@app.get("/queue_length")
async def queue_length() -> queue_info:
    """
    get internal queue length and maximum queue of the worker

    example output:
    `    {
        "queue_length": 2,
        "max_queue_length": 10
        }
    `
    """
    return queue_info(
        queue_length=internal_queue.qsize(), max_queue_length=config["maximum_queue"]
    )


if __name__ == "__main__":

    config = parse_arguments()

    # pin the model into memory

    model = load_model(config["models"], config["cuda_device"])

    # local data

    # TODO! audio_file and srt_file need to be purged periodically
    # create thread function to purge it

    audio_file = {}
    internal_queue = Queue(maxsize=config["maximum_queue"])


    process_audio_thread = Thread(target=process_audio)
    purging_thread = Thread(target=purge_data_periodially)

    process_audio_thread.start()
    purging_thread.start()

    uvicorn.run(app, port=config["port"], host=config["host"])
    pass