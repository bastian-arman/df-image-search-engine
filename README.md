# README #

## What is this repostory for? ##
This repository demonstrates a Proof of Concept (PoC) for an advanced Image Similarity Search Engine tailored for private datasets. Designed to search and retrieve visually similar images, this project leverages state-of-the-art models to efficiently index and query images, making it ideal for applications in fields such as media management, content moderation, and digital asset organization.


## Project Structures ##
```
script\                     # Shell script folder.
├──run_server.sh            # Shell script for starting streamlit server.
├──init.sh                  # Shell script for project initialize.
├──setup.sh                 # Shell script for installing project setup.
src\                        # Root folder.
├──main.py                  # Stored of main streamlit application.
├──secret.py                # Stored of all secret on .env.
│   ├──database\            # Stored of databases connection.
│   │   ├──connection.py    # Stored of database connection.
│   ├──schema\              # Stored of data format serialized by pydantic.
│   │   ├──format_data.py   # Pydantic response model.
│   ├──utils\               # Stored of list utilites based on project needs.
│   │   ├──corpus.py        # Stored of initial corpus for image tagging.
pyproject.toml              # Stored of all library based on project requirement.
```

# Project Setup Instructions

This project is developed with Python v3.12.3. To get started, you'll need to install Docker and Poetry.

## Prerequisites

- **Python v3.12.3**
- **Docker**
- **Poetry**

## Setup steps

1. **Run the setup script**
    ```
    sh scripts/setup.sh
    ```

2. **Start docker containers**
    ```
    docker-compose up
    ```

3. **Start the server script**
    ```
    sh scripts/run_server.sh
    ```

## Notes
- Ensure that docker-compose are running on your system.
- The setup.sh script configures the virtual environment and installs all necessary dependencies.
- The run_server.sh script starts the uvicorn server in debug mode for development purposes.
according to the business processes.

# Repo Owner? #
* Bastian Armananta