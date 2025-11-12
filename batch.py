import sys
import os
from subprocess import Popen
from typing import List, Tuple


def main():
    # List all items under partnet-mobility-dataset
    models_dir = 'partnet-mobility-dataset'
    models = os.listdir(models_dir)

    # Create list of tuples (index, model_name)
    indexed_models: List[Tuple[int, str]] = list(enumerate(models))

    # Declare devices list with pci- prefix
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

    # Process models with concurrency control
    processes = []
    for idx, model_name in indexed_models:
        # Use modulo to cycle through devices
        device_idx = idx % len(devices)

        # Manage concurrent processes
        while len(processes) >= 256:
            # Clean up completed processes
            for p in processes.copy():
                if p.poll() is not None:
                    processes.remove(p)

        # Prepare environment variables
        env = os.environ.copy()
        env['MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE'] = '1'
        env['DRI_PRIME'] = devices[device_idx]

        # Start the process
        command = ['python', 'demo.py', model_name]
        p = Popen(command, env=env, stdout=sys.stdout)
        processes.append(p)

    # Wait for remaining processes to complete
    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()