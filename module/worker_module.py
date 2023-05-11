import os
import configparser
import json
import time
from inference_module import *
import pika
import requests


def consumer_queue(whisper_config: str, json_config: str) -> None:
    # Get the absolute path to the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
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

    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
    url = os.environ.get(
        "CLOUDAMQP_URL",
        config["credential"],
    )
    params = pika.URLParameters(url)

    # Set up connection parameters
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # Declare the queue to consume from
    channel.queue_declare(queue=config["queue"])

    # Define a callback function to handle incoming messages
    def callback(ch, method, properties, body):
        # Specify the folder where the file will be saved
        upload_folder = "/content/uploads"  # TODO: hoist this!

        # Create the folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)

        # decode json payloads
        payload = json.loads(body)

        # download it to local storage
        download_time_start = time.time()
        os.system(f"wget -N -P {upload_folder} {payload['file_uri']}")
        download_time_stop = time.time()
        download_time = download_time_stop - download_time_start

        # put audio processing here
        file_path = os.path.join(upload_folder, payload["file_name"])
        output = whisper_to_vtt(
            model,
            file_path,
            # TODO! put this output dir in the config
            output_dir="/kaggle/working/test",
            channel_uid=[f"{payload['speaker1']}:", f"{payload['speaker2']}:"],
        )

        # save output as memory buffer data
        memory_vtt_channel1 = io.BytesIO("\n".join(output["left_channel"]).encode())
        memory_vtt_channel2 = io.BytesIO("\n".join(output["right_channel"]).encode())
        memory_vtt_conversation = io.BytesIO("\n".join(output["conversation"]).encode())

        # conversion metadata
        payload["audio_conversion_time"] = output["audio_conversion_time"]
        payload["audio_transcribtion_time"] = output["audio_transcribtion_time"]
        payload["file_name"] = {
            "speaker1": payload["file_name"].split(".")[0]
            + "_"
            + payload["speaker1"]
            + ".vtt",
            "speaker2": payload["file_name"].split(".")[0]
            + "_"
            + payload["speaker2"]
            + ".vtt",
            "conversation": payload["file_name"].split(".")[0] + "_conversation.vtt",
        }

        payload["file_uri"] = {
            "speaker1": config["bucket_vtt_url"] + payload["file_name"]["speaker1"],
            "speaker2": config["bucket_vtt_url"] + payload["file_name"]["speaker2"],
            "conversation": config["bucket_vtt_url"]
            + payload["file_name"]["conversation"],
        }

        # TODO: try catch this code bellow if this thing failed!!!
        # upload vtt files
        response_channel1 = requests.post(
            config["bucket_api_upload"],
            files={
                "file": (
                    payload["file_name"]["speaker1"],
                    memory_vtt_channel1.getvalue(),
                    "text/plain",
                )
            },
        )
        response_channel2 = requests.post(
            config["bucket_api_upload"],
            files={
                "file": (
                    payload["file_name"]["speaker2"],
                    memory_vtt_channel2.getvalue(),
                    "text/plain",
                )
            },
        )
        response_conversation = requests.post(
            config["bucket_api_upload"],
            files={
                "file": (
                    payload["file_name"]["conversation"],
                    memory_vtt_conversation.getvalue(),
                    "text/plain",
                )
            },
        )

        # should we upload metadata too?

        # ch.basic_ack(delivery_tag=method.delivery_tag)
        # ============ response message to another queue ============ #
        connection_response = pika.BlockingConnection(params)
        channel_response = connection_response.channel()

        # Declare a queue
        channel_response.queue_declare(queue=config["response_queue"])

        payload = json.dumps(payload)

        # start publishing message to rabbitMQ queue
        channel_response.basic_publish(
            exchange="", routing_key="hello_response", body=payload
        )

        channel_response.close()

        print("Received message:")
        # print(ch, method, properties)
        print(payload)

    # Start consuming messages
    channel.basic_consume(
        queue=config["queue"],
        on_message_callback=callback,
        auto_ack=True,
    )
    print("Waiting for messages. To exit, press CTRL+C")
    channel.start_consuming()


consumer_queue(whisper_config="config_file.json", json_config="conf.json")
