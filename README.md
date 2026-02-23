# URDF-Anything Dataset

The URDF-Anything dataset is hosted on Hugging Face. Download it with:

```bash
git clone https://huggingface.co/datasets/only34U/urdf-anything
```

## Renderer

A [SAPIEN](https://sapien.ucsd.edu/)-based rendering tool for generating multi-pose, multi-view images of URDF models from the PartNet-Mobility dataset.

### Installation

```bash
pip install sapien torch numpy pillow natsort
```

### Quick Start

#### Single model (demo.py)

```bash
python demo.py
```

Renders the model at `partnet-mobility-dataset/5850` by default. Output is saved to `rendered/<model_id>/`.

To render a different model, edit the `all_model_ids` list at the bottom of `demo.py`:

```python
all_model_ids = ['5850']  # replace with the target model_id
```

#### Batch rendering across multiple GPUs (batch_pool.py)

```bash
python batch_pool.py
```

Automatically enumerates all models under `partnet-mobility-dataset/` and distributes rendering tasks across the configured GPU devices in parallel.

To adjust GPU devices or per-device concurrency, edit the configuration at the top of `batch_pool.py`:

```python
devices = [
    "pci-0000_10_00_0",
    # ... fill in your actual device identifiers
]
WORKERS_PER_DEVICE = 1  # number of concurrent workers per GPU
```

