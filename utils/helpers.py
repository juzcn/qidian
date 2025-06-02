import functools
import logging
import socket
import time
from typing import Callable, Any

import requests
from flatbuffers.flexbuffers import Object

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def async_timed():
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs) -> Any:
            print(f'starting {func} with args {args} {kwargs}')
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                end = time.time()
                total = end - start
                print(f'finished {func} in {total:.4f} second(s)')

        return wrapped

    return wrapper


def dict_obj_copy(source: dict, to: Object) -> None:
    for key in source.keys():
        setattr(to, key, source[key])


def image_to_base64(url):
    response = requests.get(url)
    if response.status_code == 200:
        base64_str = base64.b64encode(response.content).decode('utf-8')
        return base64_str
    else:
        raise Exception(f"Failed to fetch image: {response.status_code}")


import base64


def file_to_base64(file):
    with open(file, 'rb') as binary_file:
        return base64.b64encode(binary_file.read()).decode('utf-8')


def sleep_forever():
    [sockA, sockB] = socket.socketpair()
    junk = sockA.recv(1)  # will never return since sockA will never receive any data


def check_url(url: str) -> bool:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        logger.error(e)
        return False
