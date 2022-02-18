#/bin/sh
export PYTHONPATH=./src:$PYTHONPATH
pytest test -x
mypy --ignore-missing-imports src
mypy --ignore-missing-imports test
