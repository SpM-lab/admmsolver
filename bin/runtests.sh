#/bin/sh
export PYTHONPATH=./src:$PYTHONPATH
pytest test
mypy --ignore-missing-imports src
mypy --ignore-missing-imports test
