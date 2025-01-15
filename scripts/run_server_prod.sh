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

# Load variables from .env
if [ -f "$PWD/.env" ]; then
    echo "Found .env file!"
    . "$PWD/.env"
else
    echo ".env File not found!"
    exit 1
fi

echo "Running streamlt server"
streamlit run src/main.py --server.port=8501 --server.address="$IP_PROD"
