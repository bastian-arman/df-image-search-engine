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
├──images\              # Directory to save image related to project.
├──tests\               # Unit testing directory.
├──script\              # Shell script directory.
│   ├──run_server.sh    # Shell script for starting streamlit server.
│   ├──init.sh          # Shell script for project initialize.
│   ├──setup.sh         # Shell script for installing project setup.
│   ├──run_test.sh      # Shell script for run all unit testing.
│   ├──mount_nas.sh     # Shell script for umounting NAS directory.
│   ├──umount_nas.sh    # Shell script for mounting NAS directory.
├──src\                 # Root project directory.
│   ├──main.py          # Stored of main streamlit application.
│   ├──secret.py        # Stored of all secret on .env.
├──utils\               # Stored of list utilites based on project needs.
│   ├──helper.py        # Stored all helper for the project.
│   ├──logger.py        # Create logging for better tracking an error.
│   ├──validator.py     # Validate user request or input.
pyproject.toml          # Stored of all library based on project requirement.
```

# Project Setup Instructions
This project is developed using WSL (Ubuntu 24.04.1 LTS on Windows 10) with Python v3.12.3. To get started, you'll need to install Docker and Poetry.

## Prerequisites

- **Python v3.12.3**
- **Docker**
- **Poetry**

## Setup steps

1. **Run setup script**
    ```
    sh scripts/setup.sh
    ```

2. **Mount NAS directory**
    ```
    sh scripts/mount_nas.sh
    ```

2. **Start the RabbitMQ container**
    ```
    docker-compose up -d
    ```

3. **Start the streamlit server**
    ```
    sh scripts/run_server.sh
    ```

4. **Access steamlit server**
    ```
    http://localhost:8501/
    ```

# Project Features
- [x] Search similar images by uploading an image (image based).
- [x] Search similar images by describing an image (text based).
- [x] Auto encode when user uploading new data (10 image) into NAS.

# Project Development
- [x] Develop at least 5 test case for all function.
- [ ] Add queueing method (e.g: Kafka, RabbitMQ, Redis) / rate limitter for preventing DDoS attack.

# Project Future Plan
- [ ] Change the current framework into more propper framework (e.g: FastAPI, Flask, Django), resulting a service-based Image Search Engine API and integrate into front-end website.
- [ ] Integrate project with database with classification method for each image and hopefully increasing an accuracy for returning similar image.


## Notes
- The setup.sh script configures the virtual environment and installs all necessary dependencies.
- Ensure that you already mounted the NAS directory using mount_nas.sh.
- The run_server.sh script starts the streamlit server.
- The run_test.sh script starts the unit testing and generates the report of test.
according to the business processes.

# Repo Owner? #
* Bastian Armananta
