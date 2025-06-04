# Server Inference Database Connector with YOLO Pose Estimation

This Python script (`db_infer_listen.py`) connects to a PostgreSQL database to retrieve image records, performs pose estimation using a YOLOv8 model, and is intended to eventually store the results back in the database.

## Core Functionality

- Connects to a PostgreSQL database using credentials from a `.env` file.
- Fetches records containing image data (`input_data` as `bytea`) and `mime_type`.
- Loads images into memory using Pillow (PIL).
- Performs batch inference on these images using a pre-trained YOLO pose estimation model (`.pt` file) via the `ultralytics` library.
- Includes placeholder logic for processing inference results (e.g., extracting keypoints).

## Setup Instructions for Ubuntu Server

Follow these steps to set up a Python environment and run the script on an Ubuntu server.

### 1. Update Package List
Ensure your package list is up-to-date:
```bash
sudo apt update
```

### 2. Install Python and pip
Most modern Ubuntu versions come with Python 3. You can verify this by typing `python3 --version`. If it's not installed, or if you need pip (Python's package installer), install them:
```bash
sudo apt install python3 python3-pip python3-venv -y
```
This command installs Python 3, pip for Python 3, and the `venv` module for creating virtual environments.

### 3. Create a Project Directory (Optional)
If you haven't already, create a directory for your project and navigate into it:
```bash
# mkdir my_project
# cd my_project
```
Place `db_infer_listen.py` inside this directory. Also, ensure your YOLO model file (e.g., `HB-eyes-400_small.pt`) is accessible, for example, in a `./models/` subdirectory.

### 4. Create a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project-specific dependencies.
```bash
python3 -m venv venv_yolo_pg
```
This creates a virtual environment named `venv_yolo_pg` in your project directory.

### 5. Activate the Virtual Environment
Before installing dependencies or running your script, activate the virtual environment:
```bash
source venv_yolo_pg/bin/activate
```
Your shell prompt should change to indicate that the virtual environment is active (e.g., `(venv_yolo_pg) user@host:...$`).

### 6. Install Required Python Packages
Install the necessary Python libraries using pip:
```bash
pip install psycopg2-binary python-dotenv Pillow ultralytics
```
- `psycopg2-binary`: A PostgreSQL adapter for Python.
- `python-dotenv`: Loads environment variables from a `.env` file.
- `Pillow`: A Python Imaging Library for opening, manipulating, and saving many different image file formats.
- `ultralytics`: The library used for YOLO model training and inference. It includes PyTorch as a dependency.

### 7. Create a `.env` File for Database Credentials
The script `db_infer_listen.py` expects the PostgreSQL password to be stored in an environment variable `PG_PASS`.

Create a file named `.env` in the same directory as the script with the following content:
```env
PG_PASS='your_actual_database_password'
```
Replace `your_actual_database_password` with your actual PostgreSQL password. Add `.env` to your `.gitignore` file if using version control.

### 8. YOLO Model Configuration
The script expects a YOLO model file. Configure its path in `db_infer_listen.py`:
```python
MODEL_PATH = "./models/HB-eyes-400_small.pt" # Adjust if your model path is different
```
Ensure this path points to your trained `.pt` model file.

### 9. Ensure PostgreSQL is Running and Accessible
Make sure your PostgreSQL server is:
- Running.
- Accessible from the Ubuntu server (check networking, firewall rules).
- Configured with a database named `base` (or modify `db_name` in the script).
- Has a user `postgres` with the password specified in `PG_PASS` (or modify `db_user` in the script).
- Has the table `server_inference` with columns like `server_infer_id`, `input_data` (bytea), and `mime_type`.

### 10. Run the Script
You can now run the Python script:
```bash
python db_infer_listen.py
```
The script will load the model, fetch records, perform batch inference, and print basic results. The `test_ids` list in the script can be modified for testing with specific records. The `BATCH_SIZE` constant can also be adjusted based on system resources.

### 11. Deactivate the Virtual Environment (When Done)
Once you are finished working, you can deactivate the virtual environment:
```bash
deactivate
```

This README provides instructions for setting up and running the script. 