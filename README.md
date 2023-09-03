# Demo: Stable Diffusion

This is a demo of the stable diffusion algorithm. Running locally with python pulling the model from hugging face.

## Setup

```bash
cd app
pyenv install 3.10.8
poetry env use $HOME/.pyenv/versions/3.10.8/bin/python
poetry install
poetry run python3 main.py
```

## References

- https://huggingface.co/docs/diffusers/optimization/mps
