""" Module for the generation of the responses, Workloads in the task queue are processed by the generation thread """
from queue import Queue
from ..iter_vllm import ItervLLM

def generate_thread_wrapper(generator: ItervLLM, task_queue: Queue, gen_config: dict):
    """ Wrapper for the generation thread """
    while True:
        inputs = task_queue.get()
        if inputs is None:
            break
        generator.generate_iter(
            **inputs,
            **gen_config
        )
