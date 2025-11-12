import os
import sys
import threading
import time
from queue import Queue
from subprocess import Popen, PIPE

devices = [
    "pci-0000_10_00_0",
    "pci-0000_16_00_0",
    "pci-0000_49_00_0",
    "pci-0000_4d_00_0",
    "pci-0000_8a_00_0",
    "pci-0000_8f_00_0",
    "pci-0000_c6_00_0",
    "pci-0000_ca_00_0"
]

WORKERS_PER_DEVICE = 1


def worker_process(device_idx: int, task_queue: Queue) -> None:
    """Worker process that maintains a persistent demo.py instance"""
    env = os.environ.copy()
    env.update({
        'MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE': '1',
        'DRI_PRIME': devices[device_idx],
        'CUDA_VISIBLE_DEVICES': str(device_idx)
    })

    # Start persistent worker process
    with Popen(
            [sys.executable, '-u', '-c',
             'from demo import ModelRenderer; \n'
             'ModelRenderer().handle_stdin()'],
            env=env,
            stdin=PIPE,
            stdout=sys.stdout,
            universal_newlines=True,
            bufsize=1
    ) as proc:
        # Feed tasks from the shared queue
        while True:
            try:
                model = task_queue.get()
                if model is None:  # Shutdown signal
                    task_queue.task_done()
                    break
                proc.stdin.write(model + '\n')
                proc.stdin.flush()
                task_queue.task_done()
            except Exception as e:
                print(f"Error: {e}")
                break
        proc.stdin.close()


def main():
    # Load models and devices
    models = os.listdir('partnet-mobility-dataset')

    # Create shared task queue
    task_queue = Queue()

    # Start worker threads
    workers = []
    for _ in range(WORKERS_PER_DEVICE):
        for device_idx in range(len(devices)):
            t = threading.Thread(
                target=worker_process,
                args=(device_idx, task_queue),
                daemon=True
            )
            t.start()
            workers.append(t)
            time.sleep(0.02)

    # Add all models to the queue
    for model in models:
        task_queue.put(model)

    # Wait for all tasks to complete
    task_queue.join()

    # Signal workers to exit
    for _ in devices:
        for _ in range(WORKERS_PER_DEVICE):
            task_queue.put(None)

    # Wait for workers to finish
    for t in workers:
        t.join()


if __name__ == '__main__':
    main()