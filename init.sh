uv init
uv run --with jupyter jupyter lab

# Within a notebook, you can import your project's modules as you would in any other file in the project. For example, if your project depends on requests, import requests will import requests from the project's virtual environment.
uv add pandas numpy matplotlib prophet scikit-learn
