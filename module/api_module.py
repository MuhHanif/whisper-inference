# main api function using fast api
from fastapi import FastAPI, File, UploadFile, HTTPException
from queue import Queue
import os
import configparser
import json
import time
from .inference_module import *
import uuid
from threading import Thread
import boto3
from botocore.client import Config
from pydantic import BaseModel


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


print(os.getcwd())
# TODO! change this to proper path
with open("config_file.json") as f:
    config = json.load(f)

with open("minio_creds.json") as creds_file:
    creds = json.load(creds_file)

app = FastAPI()

# pin the model into memory
if config["debug_without_model"] == False:
    model = load_model(config["models"], config["cuda_device"])

# local data

# TODO! audio_file and srt_file need to be purged periodically
# create thread function to purge it

audio_file = {}
internal_queue = Queue(maxsize=config["maximum_queue"])


def upload_to_bucket(file_path: str) -> None:
    """
    Uploads a file to an S3 bucket using the specified file path.

    Args:
        file_path (str): The local file path of the file to upload.

    """

    # Specify the profile name from the .aws/credentials file
    # profile_name = 's3fs-minio'
    profile_name = config["bucket_cred_profile"]

    # Create a session using the specified profile
    # TODO! cant use profile, pass as args instead
    session = boto3.session.Session()  # boto3.Session(profile_name=profile_name)

    # Create an S3 client using the session
    s3_client = session.client(
        "s3",
        endpoint_url=creds["endpoint"],
        aws_access_key_id=creds["access_key_id"],
        aws_secret_access_key=creds["secret_access_key"],
        config=Config(signature_version="s3v4"),
    )

    # Specify the bucket name
    # bucket_name = "transcribed-stash"
    bucket_name = config["bucket_name"]

    try:
        file_name = os.path.basename(file_path)

        # Specify the local file path and desired S3 key (file name)
        local_file_path = file_path
        s3_key = file_name

        # Upload the file to the bucket
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print("File uploaded successfully.")
    except Exception as error_message:
        print(error_message)

    pass


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

                    # delete VTT
                    if config["upload_to_bucket"]:
                        for file in value["list_file_path"]:
                            os.remove(file)
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
        # TODO! file need to be purged periodically or once the task is finished
        file_path = audio_file[uuid]["file"]
        # switch status to processed to indicate if this file is currently being processed
        audio_file[uuid]["status"] = "processing"
        # put try catch block here so the thread wont fail
        try:
            # raise ValueError("manually raised failed transcribtion")
            # this function call return transcribtion as dict and write it to disk
            # TODO! file need to be purged periodically or once the task is finished
            output = whisper_to_vtt(
                model=model,
                audio_file_path=file_path,
                output_dir=config["output_vtt_dir"],
                save_output_as_file=config["upload_to_bucket"],
            )
            # upload all vtt to s3 bucket
            if config["upload_to_bucket"]:
                for upload_file_path in output["list_file_path"]:
                    upload_to_bucket(upload_file_path)
                # store this in dict so purging script can get the file easily
                audio_file[uuid]["list_file_path"] = output["list_file_path"]

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


process_audio_thread = Thread(target=process_audio)
purging_thread = Thread(target=purge_data_periodially)

process_audio_thread.start()
purging_thread.start()


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
    # return {
    #     "queue_length": internal_queue.qsize(),
    #     "max__queue_length": config["maximum_queue"],
    # }
    return queue_info(
        queue_length=internal_queue.qsize(), max_queue_length=config["maximum_queue"]
    )
