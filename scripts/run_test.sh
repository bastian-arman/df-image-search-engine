#!/bin/bash

echo "Checking OS Environment"
if grep -qEi "(Microsoft|WSL)" /proc/version &>/dev/null; then
  echo "WSL detected"
  . .venv/bin/activate
else
  case "$OSTYPE" in
    linux*)
      echo "Linux based OS detected"
      . .venv/bin/activate
      ;;
    cygwin* | msys* | mingw*)
      echo "Windows based OS detected"
      source .venv/Scripts/activate
      ;;
    *)
      echo "Unsupported OS detected. This feature is not developed yet."
      exit 1
      ;;
  esac
fi

echo "Running tests"
coverage run -m --source=tests/ pytest tests/

echo "Generating coverage report"
coverage report -m --skip-empty
coverage html

echo "Test finished"
