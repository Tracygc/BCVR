All dependencies are tracked in `pyproject.toml` and no new files should be added to
the `requirements` directory.

We maintain `base.txt` to allow installing the package only with the dependencies
necessary to use the API part of the package. The package can be installed with API
only dependencies by running:
```
pip install -r requirements/base.txt
pip install lightly --no-deps
```
