import nsq
from typing import List, Dict


def read_nsq(queue_topic: str, nsqd_tcp_addresses: List[str]) -> Dict[str, List[str]]:
    """
    Reads messages from an NSQ queue and returns them as a dictionary.

    Args:
        queue_topic: The name of the NSQ topic to read from.
        nsqd_tcp_addresses: A list of one or more nsqd TCP addresses to connect to NSQD producers.

    Returns:
        A dictionary with a single key 'messages' containing a list of messages received.

    """
    messages = []

    def handler(message):
        messages.append(message.body.decode())
        return True

    r = nsq.Reader(
        message_handler=handler,
        nsqd_tcp_addresses=nsqd_tcp_addresses,
        topic=queue_topic,
        channel="my_channel",
    )
    nsq.run()

    return {"messages": messages}


print()
