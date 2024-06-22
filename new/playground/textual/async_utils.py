import asyncio
from asyncio import AbstractEventLoop
from threading import Thread
from queue import Queue as ThreadingQueue

def _threading_queue_item_ready(semaphore):
    # increase the count of available items in the queue so
    # that it can be read in an async-friendly way:
    semaphore.release()

def _producer_wrapper(gen, loop: AbstractEventLoop, queue, semaphore):
    def wrapper(*args, **kwargs):
        for item in gen:
            queue.put(item)
            loop.call_soon_threadsafe(_threading_queue_item_ready, semaphore)
        queue.put(None)
        loop.call_soon_threadsafe(_threading_queue_item_ready, semaphore)
    wrapper()

async def async_gen(sync_gen):
    """wrapper for the "generator asynchronization" mechanism"""
    semaphore = asyncio.Semaphore(0)
    queue = ThreadingQueue()
    producer_thread = Thread(
        target=_producer_wrapper,
        args=(sync_gen, asyncio.get_running_loop(), queue, semaphore)
    )
    producer_thread.start()
    while True:
        await semaphore.acquire()
        # we only get here when there is at least one item in the queue
        item = queue.get()
        if item is None:
            break
        yield item
        # intentionaly do not "release" the semaphore: available item count
        # is increased by the producer code
