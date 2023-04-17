import nsq
from typing import List, Dict

def read_nsq(queue_topic: str, lookupd_http_addresses: List[str]) -> Dict[str, List[str]]:
    """
    Reads messages from an NSQ queue and returns them as a dictionary.

    Args:
        queue_topic: The name of the NSQ topic to read from.
        lookupd_http_addresses: A list of one or more lookupd HTTP addresses to discover NSQD producers.

    Returns:
        A dictionary with a single key 'messages' containing a list of messages received.

    """
    messages = []

    def handler(message):
        messages.append(message.body.decode())
        return True

    r = nsq.Reader(
        message_handler=handler,
        lookupd_http_addresses=lookupd_http_addresses,
        topic=queue_topic,
        channel='my_channel'
    )
    nsq.run()

    return {'messages': messages}