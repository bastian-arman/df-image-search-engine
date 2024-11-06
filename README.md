# README #

## What is this repostory for? ##
This repository demonstrates a Proof of Concept (PoC) for an advanced Image Similarity Search Engine tailored for private datasets. Designed to search and retrieve visually similar images, this project leverages state-of-the-art models to efficiently index and query images.

## Approach

| ![CLIP](https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/CLIP.png) |
|:--:|
| Image Credit: https://github.com/openai/CLIP |

## Architecture Flow
| ![CLIP](https://github.com/bastian-arman/df-image-search-engine/blob/feature/global-nas/images/Architecture%20Flow.png) |
|:--:|
| Image Credit: https://github.com/bastian-arman/df-image-search-engine |

## Project Structures ##
```
script\                         # Shell script folder.
├──run_server.sh                # Shell script for starting streamlit server.
├──init.sh                      # Shell script for project initialize.
├──setup.sh                     # Shell script for installing project setup.
src\                            # Root folder.
├──main.py                      # Stored of main streamlit application.
├──secret.py                    # Stored of all secret on .env.
│   ├──utils\                   # Stored of list utilites based on project needs.
│   │   ├──helper.py            # Stored all helper for the project.
│   │   ├──logger.py            # Create logging for better tracking an error.
│   │   ├──nas_connection.py    # Integrate local streamlit server into NAS server.
pyproject.toml                  # Stored of all library based on project requirement.
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

4. **Access steamlit server**
    ```
    http://localhost:8501/
    ```

## Notes
- Ensure that docker-compose are running on your system.
- The setup.sh script configures the virtual environment and installs all necessary dependencies.
- The run_server.sh script starts the streamlit server.
according to the business processes.

# Repo Owner? #
* Bastian Armananta
