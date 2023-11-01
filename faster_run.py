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
from faster_whisper import WhisperModel
import subprocess
import http.client
import ast

app = FastAPI()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse JSON configuration into a dictionary")
    
    # Add arguments for each field in the JSON
    parser.add_argument("--models", type=str, default='medium', help="Specify the model name (e.g., 'medium', 'large-v2'). default: medium")
    # parser.add_argument("--parallel_device",
    #     nargs='+',  # This tells argparse to accept one or more values as a list
    #     help="Specify the model names (e.g., 'medium', 'large-v2')."
    # )
    # parser.add_argument("--device", type=str, default='cuda', help="Specify the inference device (e.g., 'cuda', 'cpu'). default: 'cuda'")
    parser.add_argument("--device", type=str, default='cuda', help="Specify the inference device (e.g., 'cuda', 'cpu'). default: 'cuda'")
    parser.add_argument("--num_workers", type=int, default=5, help="Specify the ammount of concurent thread that's running on the gpu. default: 1")
    parser.add_argument("--maximum_queue", type=int, default=10, help="Specify the maximum queue size. default: 10")
    parser.add_argument("--output_vtt_dir", type=str, required=True, help="Specify the output VTT directory, required for caching process")
    parser.add_argument("--audio_folder", type=str, required=True, help="Specify the audio folder directory, required for caching process")
    parser.add_argument("--ephemeral_data_duration", type=int, default=500, help="Specify the ephemeral data duration before deleting it. default: 500")
    parser.add_argument("--ephemeral_check_period", type=int, default=5, help="Specify the ephemeral check period to check if the transcription is over the ephemeral_data_duration. default: 5")
    parser.add_argument("--port", type=int, default=7777, help="port number. default: 7777")
    parser.add_argument("--host", type=str, default="localhost", help="host ip. default: 'localhost'")

    args = parser.parse_args()
    
    return args


def load_model(model_name: str, device: str = "cuda", num_workers: int = 5):
    """
    Loads a Whisper model to GPU

    BUG! when defining cuda device. whisper always tried to load to cuda:0!
    probably problem with load_model method

    Args:
        model_name (str): The name of the model to load.
        device (str): CUDA device to run on.
        num_workers (int): number of worker threads
    Returns:
        whisper.model: The loaded Whisper model.
    """
    # Load the Whisper model

    tic = time.time()
    model = WhisperModel(
        model_name, 
        device=device, 
        compute_type="float16", 
        device_index=[0,1],
        cpu_threads=num_workers,
        num_workers=num_workers,
        )
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


class summarize_gpt(BaseModel):
    gpt_worker_ip: str
    gpt_worker_port: str
    gpt_worker_endpoint: str
    instruction: str = None
    audio_file_url: str
    token_limit: int


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
                    > config.ephemeral_data_duration
                ):
                    # deleting audio file
                    os.remove(value["file"])

                    # delete dict
                    del audio_file[key]

                # delete failed transcribtion after x seconds
                if (
                    value["status"] == "failed to transcribe audio"
                    and time.time() - value["timestamp"]
                    > config.ephemeral_data_duration
                ):
                    # deleting audio file
                    os.remove(value["file"])
                    # delete dict
                    del audio_file[key]

        except Exception as purging_error:
            print(purging_error)

        # sleep periodically to not hog resources
        time.sleep(config.ephemeral_check_period)


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
            output = fast_whisper_to_vtt(
                model=model,
                audio_file_path=file_path,
                output_dir=config.output_vtt_dir,
                save_output_as_file=False,
            )  

            # do GPT sumarization with instruction
            if "GPT_instruction" in audio_file[uuid]:
                gpt_start = time.time()
                audio_file[uuid]["GPT_endpoint"]
                conn = http.client.HTTPConnection(audio_file[uuid]["GPT_ip"], audio_file[uuid]["GPT_port"])

                headersList = {
                    "Accept": "*/*",
                    "User-Agent": "Thunder Client (https://www.thunderclient.com)",
                    "Content-Type": "application/json" 
                }
                
                if "right_channel" in output:
                    combined_channel = ast.literal_eval(repr(output["conversation"]))
                else:
                    combined_channel = ast.literal_eval(repr(output["left_channel"]))
                payload = json.dumps({
                    "model": "zephyr-7b", 
                    "prompt": f'<|system|></s><|user|>{audio_file[uuid]["GPT_instruction"]}{combined_channel}', 
                    "temperature": 0.7, 
                    "repetition_penalty": 1.0, 
                    "top_p": 0.9, 
                    "max_new_tokens": audio_file[uuid]["token_limit"], 
                    "stop": "\n[INST]", 
                    "stop_token_ids": None, 
                    "echo": False
                    
                })

                conn.request("POST", audio_file[uuid]["GPT_endpoint"] , payload, headersList)
                response = conn.getresponse()
                result = response.read()
                audio_file[uuid]["GPT_output"]=json.loads(result)
                gpt_stop = time.time()
                audio_file[uuid]["GPT_auto_sumarization_time"] = gpt_stop-gpt_start

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
    upload_folder = config.audio_folder

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


@app.post("/download_audio_to_queue_process/")
async def download_audio_to_queue_process(url: str) -> upload_audio:
    """
    Endpoint that accepts an audio file URL and push it to internal queue to be processed

    will return queue id that can be used on `/transcribtion_status/` endpoint to retrieve transcription

    Arg:
        url: Audio file url must have this file extension `.mp3`, `.ogg`, `.wav`.

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
    upload_folder = config.audio_folder

    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Check if the uploaded file is an audio file
    valid_extensions = [".mp3", ".wav", ".ogg"]
    file_extension = os.path.splitext(url)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="File must be an audio file")

    uuid_name = str(uuid.uuid4())

    # Save the file to the specified folder
    file_path = os.path.join(upload_folder, f"{uuid_name}{file_extension}")

    command = f"wget -O {file_path} {url}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

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


@app.post("/download_transcribe_sumarize_wrapper/")
async def download_transcribe_sumarize_wrapper(item: summarize_gpt) -> upload_audio:
    """
    Endpoint that accepts an audio file URL and push it to internal queue to be processed

    will return queue id that can be used on `/transcribtion_status/` endpoint to retrieve transcription

    Arg:
        url: Audio file url must have this file extension `.mp3`, `.ogg`, `.wav`.

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
    upload_folder = config.audio_folder

    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Check if the uploaded file is an audio file
    valid_extensions = [".mp3", ".wav", ".ogg"]
    file_extension = os.path.splitext(item.audio_file_url)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="File must be an audio file")

    uuid_name = str(uuid.uuid4())

    # Save the file to the specified folder
    file_path = os.path.join(upload_folder, f"{uuid_name}{file_extension}")

    command = f"wget -O {file_path} {item.audio_file_url}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # put process in queue
    internal_queue.put(uuid_name)

    # store necessary data to be processed
    audio_file[uuid_name] = {}
    audio_file[uuid_name]["queue_id"] = uuid_name
    audio_file[uuid_name]["file"] = file_path
    # status can be (stored or processing)
    audio_file[uuid_name]["status"] = "stored"
    # time is required for other function to periodicaly check if
    # this task is being 
    audio_file[uuid_name]["GPT_instruction"] = item.instruction
    audio_file[uuid_name]["GPT_endpoint"] = item.gpt_worker_endpoint
    audio_file[uuid_name]["GPT_ip"] = item.gpt_worker_ip
    audio_file[uuid_name]["GPT_port"] = item.gpt_worker_port
    audio_file[uuid_name]["token_limit"] = item.token_limit
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
        queue_length=internal_queue.qsize(), max_queue_length=config.maximum_queue
    )


if __name__ == "__main__":

    config = parse_arguments()

    # pin the model into memory

    model = load_model(config.models, config.device, config.num_workers)

    # local data

    # TODO! audio_file and srt_file need to be purged periodically
    # create thread function to purge it

    audio_file = {}
    internal_queue = Queue(maxsize=config.maximum_queue)


    process_audio_thread = Thread(target=process_audio)
    purging_thread = Thread(target=purge_data_periodially)

    process_audio_thread.start()
    purging_thread.start()

    uvicorn.run(app, port=config.port, host=config.host)
    pass