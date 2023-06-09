import io
from typing import List, Tuple, Union, Dict
import numpy as np
import soundfile as sf
import whisper
import time
import torch
import pandas as pd
import os
import subprocess
import shutil
import requests
import json
from urllib.parse import urljoin


class transcriptionError(Exception):
    pass


def scale_audio(audio: np.ndarray) -> np.ndarray:
    """
    Scales audio data to be within -1 and 1.

    Args:
        audio (np.ndarray): Input audio data.

    Returns:
        np.ndarray: Scaled audio data.
    """
    max_sample = np.max(np.abs(audio))
    if max_sample > 0.8:
        audio = audio / max_sample
    return audio


def split_stereo_audio(audio_data: Union[str, bytes]) -> List[np.ndarray]:
    """Splits a stereo audio file in either file or byte format into two separate mono audio files and stores them in a list.

    Args:
        audio_data (Union[str, bytes]): The stereo audio file, either in file path string or byte format.

    Raises:
        ValueError: If the input audio is not in stereo format.

    Returns:
        List[np.ndarray]: A list containing two numpy arrays, one for each mono channel. # change this
    """
    # If the input is a string, assume it's a file path
    if isinstance(audio_data, str):
        # Load the audio file from disk
        audio, sample_rate = sf.read(audio_data)
    # Otherwise, assume it's a bytes object
    else:
        # Load the audio file from bytes
        audio, sample_rate = sf.read(io.BytesIO(audio_data))
    # Check if audio is stereo
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Input file is not stereo audio")

    # Split audio into left and right channels
    left = audio[:, 0].astype("float32")  # scale_audio(audio[:, 0].astype("float32"))
    right = audio[:, 1].astype("float32")  # scale_audio(audio[:, 1].astype("float32"))

    # Store left and right channels in a list
    mono_audio = {
        "Caller: ": left,
        "Reciever: ": right,
    }  # [left, right] np.stack().astype("float32")

    return mono_audio


def split_stereo_audio_ffmpeg(input_file: str) -> Dict[str, str]:
    """Split an audio file into separate left and right channels using ffmpeg.

    Args:
        input_file (str): The path to the input audio file.

    Returns:
        A dictionary with the following keys:
            'left': The path to the left channel output file.
            'right': The path to the right channel output file.
    """

    # stereo check
    channel_count = [
        "ffprobe",
        "-i",
        f"{input_file}",
        "-show_entries",
        "stream=channels",
        "-select_streams",
        "a:0",
        "-of",
        "compact=p=0:nk=1",
        "-v",
        "0",
    ]
    result = subprocess.run(channel_count, stdout=subprocess.PIPE)
    numb_of_channels = int(result.stdout.decode("utf-8")[0])

    if numb_of_channels != 2:
        raise ValueError("Input file is not stereo audio")

    # Determine input and output file extensions
    input_ext = os.path.splitext(input_file)[1]

    folder_name = "/tmp/whisper_temp"
    # temporary dir to store output
    try:
        folders = os.makedirs(folder_name)
    except FileExistsError:
        pass

    # Determine output file names
    left_file = os.path.join(folder_name, "left" + input_ext)
    right_file = os.path.join(folder_name, "right" + input_ext)

    # Build ffmpeg command
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-map_channel",
        "0.0.0",
        left_file,
        "-map_channel",
        "0.0.1",
        right_file,
    ]

    # Run ffmpeg command and capture output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check for errors
    if result.returncode != 0:
        raise Exception(result.stderr.decode("utf-8"))

    # Return output file names as dictionary
    return {"Caller: ": left_file, "Reciever: ": right_file}


def whisper_inference(
    model, audio_file: str, channel_uid: list
) -> Tuple[List[str], float]:
    """
    Uses the OpenAI Whisper model to transcribe stereo audio for each channel.
    TODO!: dockstring incomplete and wrong

    Args:
        model: An instance of the Whisper model loaded using `whisper.load_model`.
        audio_file (str): The stereo audio file path.
        channel_uid (list): [left/speaker1, right/speaker2].

    Returns:
        A tuple containing a list of transcribed strings (one for each audio channel) and the time taken for the conversion process.

    Raises:
        ValueError: If the input audio is not in stereo format.

    Example:
        >>> transcriptions, conversion_time = whisper_inference('model.pth', 'audio.wav')
        >>> print(transcriptions)
        ['Transcription for left channel', 'Transcription for right channel']
        >>> print(conversion_time)
        3.14159
    """
    start_audio_processing_time = time.time()
    # Split the stereo audio into left and right channels
    split_data = split_stereo_audio_ffmpeg(audio_file)
    # Perform the transcription for each audio channel while measuring conversion time
    stop_audio_processing_time = time.time()
    audio_processing_time = stop_audio_processing_time - start_audio_processing_time

    start_transcribtion_time = time.time()
    # [left, right]
    transcriptions = []
    count = 0
    for channel, data in split_data.items():
        # swap this with api
        results = model.transcribe(data, language="en", fp16=False, temperature=0)
        results = pd.DataFrame(results["segments"])
        # prepend receiver and caller
        results["text"] = channel_uid[count] + results["text"]
        results["channel"] = 0 if channel == "Reciever: " else 1
        transcriptions.append(results)
        count = count + 1
    stop_transcribtion_time = time.time()
    transcribtion_time = stop_transcribtion_time - start_transcribtion_time

    # create a back and forth conversation order
    combined_transcription = pd.concat(transcriptions)
    combined_transcription = combined_transcription.sort_values(
        by=["start", "seek", "channel"]
    )
    combined_transcription = combined_transcription.reset_index(drop=True)

    return (
        transcriptions,
        combined_transcription,
        transcribtion_time,
        audio_processing_time,
    )


def create_srt(
    df: pd.DataFrame,
    start_col: str = "start_speech",
    end_col: str = "end_speech",
    subtitle_col: str = "subtitles",
) -> str:
    """
    Create an SRT file from a pandas dataframe with columns specified by the `start_col`, `end_col`, and `subtitle_col`
    parameters, and return the file contents as a string.

    Parameters:
        df (pd.DataFrame): The pandas dataframe containing the subtitles data.
        start_col (str): The name of the dataframe column containing the start time of the subtitles. Default is "start_speech".
        end_col (str): The name of the dataframe column containing the end time of the subtitles. Default is "end_speech".
        subtitle_col (str): The name of the dataframe column containing the subtitle text. Default is "subtitles".

    Returns:
        str: The SRT file contents as a string.
    """

    srt_content = ""
    for i, row in df.iterrows():
        # Write the index number
        srt_content += f"{i+1}\n"

        # Write the time range
        start_time = pd.to_datetime(row[start_col], unit="s").strftime("%H:%M:%S,%f")[
            :-3
        ]
        end_time = pd.to_datetime(row[end_col], unit="s").strftime("%H:%M:%S,%f")[:-3]
        srt_content += f"{start_time} --> {end_time}\n"

        # Write the subtitle text
        srt_content += f"{row[subtitle_col]}\n"
        srt_content += "\n"

    return srt_content.strip()


def create_vtt(
    df: pd.DataFrame,
    start_col: str = "start_speech",
    end_col: str = "end_speech",
    subtitle_col: str = "subtitles",
) -> str:
    """
    Create a VTT file from a pandas dataframe with columns specified by the `start_col`, `end_col`, and `subtitle_col`
    parameters, and return the file contents as a string.

    Parameters:
        df (pd.DataFrame): The pandas dataframe containing the subtitles data.
        start_col (str): The name of the dataframe column containing the start time of the subtitles. Default is "start_speech".
        end_col (str): The name of the dataframe column containing the end time of the subtitles. Default is "end_speech".
        subtitle_col (str): The name of the dataframe column containing the subtitle text. Default is "subtitles".

    Returns:
        str: The VTT file contents as a string.
    """

    vtt_content = "WEBVTT\n\n"
    for i, row in df.iterrows():
        # Write the index number
        vtt_content += f"{i+1}\n"

        # Write the time range
        start_time = pd.to_datetime(row[start_col], unit="s").strftime("%H:%M:%S,%f")[
            :-3
        ]
        end_time = pd.to_datetime(row[end_col], unit="s").strftime("%H:%M:%S,%f")[:-3]
        vtt_content += f"{start_time} --> {end_time}\n"

        # Write the subtitle text
        vtt_content += f"{row[subtitle_col]}\n"
        vtt_content += "\n"

    return vtt_content.strip()


def create_custom_vtt(
    df: pd.DataFrame,
    start_col: str = "start_speech",
    end_col: str = "end_speech",
    subtitle_col: str = "subtitles",
) -> str:
    """
    Create a custom VTT file from a pandas dataframe with columns specified by the `start_col`, `end_col`, and `subtitle_col`
    parameters, and return the file contents as a string.

    Parameters:
        df (pd.DataFrame): The pandas dataframe containing the subtitles data.
        start_col (str): The name of the dataframe column containing the start time of the subtitles. Default is "start_speech".
        end_col (str): The name of the dataframe column containing the end time of the subtitles. Default is "end_speech".
        subtitle_col (str): The name of the dataframe column containing the subtitle text. Default is "subtitles".

    Returns:
        str: The custom VTT file contents as a string.
    """

    vtt_content = ""
    for i, row in df.iterrows():
        # Write the time range
        start_time = pd.to_datetime(row[start_col], unit="s").strftime("%H:%M:%S,%f")[
            :-3
        ]
        end_time = pd.to_datetime(row[end_col], unit="s").strftime("%H:%M:%S,%f")[:-3]
        vtt_content += f"{start_time} --> {end_time} "

        # Write the subtitle text
        vtt_content += f"{row[subtitle_col]}\n"

    return vtt_content.strip()


def whisper_to_vtt(
    model,
    audio_file_path: str,
    save_output_as_file: bool = True,
    output_dir: str = "/tmp/whisper_temp",
    channel_uid: list = ["left_channel", "right_channel"],
) -> Dict[str, str]:
    """
    Converts an audio file to three VTT (WebVTT) files containing the speech of the caller, receiver, and conversation.

    Args:
        model (whisper.model): The Whisper model.
        audio_file_path (str): The path to the audio file to be processed.
        save_output_as_file (bool, optional): Whether to save the output as files or not. Defaults to True.
        output_dir (str, optional): The directory to save the VTT files. Defaults to "/tmp/whisper_temp".
        channel_uid (list): [left/speaker1, right/speaker2] defaulted to ["left_channel", "right_channel"].

    Return:
        dict:
            'left_channel' (str): transcribed left channel,
            'right_channel' (str): transcribed right channel,
            'conversation' (str): transcribed both channel,
            'audio_conversion_time' (int): ffmpeg conversion time,
            'audio_transcribtion_time' (int): whisper inference time,
            'list_file_path'(list, optional): saved file path.

    Outputs:
        bunch of vtt files on save_output_as_files with format
            1. {audio_file_name}_{channel_uid[0]}.vtt
            2. {audio_file_name}_{channel_uid[1]}.vtt
            3. {audio_file_name}_conversation.vtt
    """

    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Perform inference
    result = whisper_inference(model, audio_file_path, channel_uid)

    # get audio name
    # /path/to/audio/file.mp3
    file_name = os.path.basename(audio_file_path)
    file_name = os.path.splitext(file_name)[0]

    # Convert inference results to VTT strings
    caller = create_custom_vtt(result[0][0], "start", "end", "text")
    receiver = create_custom_vtt(result[0][1], "start", "end", "text")
    conversation = create_custom_vtt(result[1], "start", "end", "text")

    # Save VTT files
    if save_output_as_file:
        with open(
            os.path.join(output_dir, f"{file_name}_{channel_uid[0]}.vtt"), "w"
        ) as f:
            f.write(caller)
        with open(
            os.path.join(output_dir, f"{file_name}_{channel_uid[1]}.vtt"), "w"
        ) as f:
            f.write(receiver)
        with open(os.path.join(output_dir, f"{file_name}_conversation.vtt"), "w") as f:
            f.write(conversation)

    returned_output = {
        "left_channel": caller.split("\n"),
        "right_channel": receiver.split("\n"),
        "conversation": conversation.split("\n"),
        "audio_conversion_time": result[3],
        "audio_transcribtion_time": result[2],
    }

    # append file path to make it easier to locate on down stream function
    if save_output_as_file:
        returned_output["list_file_path"] = [
            os.path.join(output_dir, f"{file_name}_{channel_uid[0]}.vtt"),
            os.path.join(output_dir, f"{file_name}_{channel_uid[1]}.vtt"),
            os.path.join(output_dir, f"{file_name}_conversation.vtt"),
        ]

    return returned_output


def move_file_to_folder(
    file_path: str, folder_path: str = "/temp/runpod_file_queue"
) -> str:
    """
    Moves a file to a specified folder and returns the new file path.

    Args:
        file_path (str): The path of the file to be moved.
        folder_path (str, optional): The path of the destination folder.
            Defaults to `/temp/runpod_file_queue`.


    Returns:
        str: The new file path after moving the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Extract the file name from the file path
    file_name = os.path.basename(file_path)

    # Create the destination folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Construct the new file path in the destination folder
    new_file_path = os.path.join(folder_path, file_name)

    # Move the file to the destination folder
    shutil.move(file_path, new_file_path)

    # Return the new file path
    return new_file_path


def convert_local_path_to_url(file_path: str, config: str) -> str:
    """
    this is dummy function relies on simplehttpserver to serve the file to runpod api
    please change this function with proper s3 bucket

    Args:
        file_path (str): The path of the file to be replaced with public url.
        config (str): json path to some credentials settings.

    Returns:
        str: publicly accessible file url.
    """

    with open(config, "r") as json_config:
        # Load the JSON data into a dictionary
        config_data = json.load(json_config)

    # move the file into publicly shared folder
    path = move_file_to_folder(
        file_path=file_path, folder_path=config_data["simple_http_server_dir"]
    )

    # get file name
    filename = os.path.basename(path)

    return urljoin(config_data["cloudflare_zero_trust_domain"], filename)


def get_transcription_from_runpod(
    api_key: str, audio_url: str, model: str = "base"
) -> requests.Response:
    """
    Retrieve transcription using the RunPod API.

    Args:
        api_key (str): The API key for authentication.
        audio_url (str): The URL of the audio file to transcribe.
        model (str, optional): Whisper model options
            `tiny`, `base`, `small`, `medium`, `large`, `large-v2` Defaults to `base`.

    Returns:
        requests.Response: The response object containing the transcription result.
    """
    output = None

    payload = {
        "input": {
            "audio": audio_url,
            "model": model,
            "transcription": "plain text",
            "translate": False,
            "temperature": 0,
            "best_of": 5,
            "beam_size": 5,
            "suppress_tokens": "-1",
            "condition_on_previous_text": False,
            "temperature_increment_on_fallback": 0.2,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
        }
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": api_key,
    }

    queue_headers = {"accept": "application/json", "authorization": api_key}

    # run whisper api url endpoint
    run_url = "https://api.runpod.ai/v2/faster-whisper/run"
    check_job_url = "https://api.runpod.ai/v2/faster-whisper/status/"

    # push transcription to api endpoint
    response = requests.post(run_url, json=payload, headers=headers)

    # if request successful periodically check the queue
    response.raise_for_status()

    # gets queue id from json response
    queue_id = response.json()["id"]

    # append job id
    check_job_url = urljoin(check_job_url, queue_id)

    # periodically check if the job is completed
    while True:
        # put sleep at top so i dont forget about it
        time.sleep(1)

        queue_headers = {"accept": "application/json", "authorization": api_key}

        # retrieve transcription status from api endpoint
        queue_response = requests.get(check_job_url, headers=queue_headers)

        # raise any request error
        queue_response.raise_for_status()

        # convert queue_response to dict
        queue_response = queue_response.json()

        # gets queue status from json response
        status = queue_response["status"]

        if status == "COMPLETED":
            output = queue_response["output"]
            break

        if status == "FAILED":
            raise transcriptionError(
                f"Runpod failed to transcribe the audio: {queue_response}"
            )
            break

        print(status)

    return output


def runpod_whisper_inference(
    model: str, audio_file: str, channel_uid: list, api_key: str, config_dir: str = None
) -> Tuple[List[str], float]:
    """
    Uses runpod Whisper model to transcribe stereo audio for each channel.

    Args:
        model (str, optional): Whisper model options
            `tiny`, `base`, `small`, `medium`, `large`, `large-v2` Defaults to `base`.
        audio_file (str): The stereo audio file path.
        channel_uid (list): [left/speaker1, right/speaker2].
        api_key (str): runpod API key.
        config_dir (str): json config path for simplehttpserver

    Note:
        replace simpleHTTPserver with proper S3 storage!

    Returns:
        A tuple containing a list of transcribed strings (one for each audio channel) and the time taken for the conversion process.

    Raises:
        ValueError: If the input audio is not in stereo format.

    """
    start_audio_processing_time = time.time()
    # Split the stereo audio into left and right channels
    split_data = split_stereo_audio_ffmpeg(audio_file)
    # Perform the transcription for each audio channel while measuring conversion time
    stop_audio_processing_time = time.time()
    audio_processing_time = stop_audio_processing_time - start_audio_processing_time

    start_transcribtion_time = time.time()
    # [left, right]
    transcriptions = []
    count = 0
    for channel, data in split_data.items():
        # swap this with api

        #########################################################################
        #                                                                       #
        #               TODO! change this with proper S3 file url!              #
        #                                                                       #
        #########################################################################

        # create public url for the audio file
        data = convert_local_path_to_url(file_path=data, config=config_dir)

        # perform an inference
        results = get_transcription_from_runpod(
            model=model, api_key=api_key, audio_url=data
        )
        results = pd.DataFrame(results["segments"])
        # prepend receiver and caller
        results["text"] = channel_uid[count] + results["text"]
        results["channel"] = 0 if channel == "Reciever: " else 1
        transcriptions.append(results)
        count = count + 1
    stop_transcribtion_time = time.time()
    transcribtion_time = stop_transcribtion_time - start_transcribtion_time

    # create a back and forth conversation order
    combined_transcription = pd.concat(transcriptions)
    combined_transcription = combined_transcription.sort_values(
        by=["start", "seek", "channel"]
    )
    combined_transcription = combined_transcription.reset_index(drop=True)

    return (
        transcriptions,
        combined_transcription,
        transcribtion_time,
        audio_processing_time,
    )


def runpod_whisper_to_vtt(
    api_key: str,
    simple_http_server_config: str,
    audio_file_path: str,
    save_output_as_file: bool = True,
    output_dir: str = "/tmp/whisper_temp",
    channel_uid: list = ["left_channel", "right_channel"],
    model: str = "base",
) -> Dict[str, str]:
    """
    Converts an audio file to three VTT (WebVTT) files containing the speech of the caller, receiver, and conversation.

    Args:
        api_key (str): runpod API key.
        simple_http_server_config (str): path to simplehttpserver JSON config.
        audio_file_path (str): The path to the audio file to be processed.
        save_output_as_file (bool, optional): Whether to save the output as files or not. Defaults to True.
        output_dir (str, optional): The directory to save the VTT files. Defaults to "/tmp/whisper_temp".
        channel_uid (list): [left/speaker1, right/speaker2] defaulted to ["left_channel", "right_channel"].
        model (str, optional): Whisper model options
            `tiny`, `base`, `small`, `medium`, `large`, `large-v2` Defaults to `base`.

    Return:
        dict:
            'left_channel' (str): transcribed left channel,
            'right_channel' (str): transcribed right channel,
            'conversation' (str): transcribed both channel,
            'audio_conversion_time' (int): ffmpeg conversion time,
            'audio_transcribtion_time' (int): whisper inference time,
            'list_file_path'(list, optional): saved file path.

    Outputs:
        bunch of vtt files on save_output_as_files with format
            1. {audio_file_name}_{channel_uid[0]}.vtt
            2. {audio_file_name}_{channel_uid[1]}.vtt
            3. {audio_file_name}_conversation.vtt
    """

    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Perform inference
    result = runpod_whisper_inference(
        model=model,
        audio_file=audio_file_path,
        channel_uid=channel_uid,
        api_key=api_key,
        config_dir=simple_http_server_config,
    )

    # get audio name
    # /path/to/audio/file.mp3
    file_name = os.path.basename(audio_file_path)
    file_name = os.path.splitext(file_name)[0]

    # Convert inference results to VTT strings
    caller = create_custom_vtt(result[0][0], "start", "end", "text")
    receiver = create_custom_vtt(result[0][1], "start", "end", "text")
    conversation = create_custom_vtt(result[1], "start", "end", "text")

    # Save VTT files
    if save_output_as_file:
        with open(
            os.path.join(output_dir, f"{file_name}_{channel_uid[0]}.vtt"), "w"
        ) as f:
            f.write(caller)
        with open(
            os.path.join(output_dir, f"{file_name}_{channel_uid[1]}.vtt"), "w"
        ) as f:
            f.write(receiver)
        with open(os.path.join(output_dir, f"{file_name}_conversation.vtt"), "w") as f:
            f.write(conversation)

    returned_output = {
        "left_channel": caller.split("\n"),
        "right_channel": receiver.split("\n"),
        "conversation": conversation.split("\n"),
        "audio_conversion_time": result[3],
        "audio_transcribtion_time": result[2],
    }

    # append file path to make it easier to locate on down stream function
    if save_output_as_file:
        returned_output["list_file_path"] = [
            os.path.join(output_dir, f"{file_name}_{channel_uid[0]}.vtt"),
            os.path.join(output_dir, f"{file_name}_{channel_uid[1]}.vtt"),
            os.path.join(output_dir, f"{file_name}_conversation.vtt"),
        ]

    return returned_output


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
    tic = time.time()
    model = whisper.load_model(model_name, device=device)
    toc = time.time()
    print(f"Model loaded, took {round(toc-tic, 2)} second(s)")

    return model
