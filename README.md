# Server Inference Database Connector

This Python script (`db_infer_listen.py`) connects to a PostgreSQL database to retrieve inference records.

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
mkdir ~/envs
cd ~/envs
```
Place `db_infer_listen.py` inside this directory.

### 4. Create a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project-specific dependencies.
```bash
python3 -m venv yolo_pg
```
This creates a virtual environment named `venv` in your project directory.

### 5. Activate the Virtual Environment
Before installing dependencies or running your script, activate the virtual environment:
```bash
source yolo_pg/bin/activate
```
Your shell prompt should change to indicate that the virtual environment is active (e.g., `(venv) user@host:...$`).

### 6. Install Required Python Packages
Install the necessary Python libraries using pip:
```bash
pip install psycopg2-binary python-dotenv
```
- `psycopg2-binary`: A PostgreSQL adapter for Python. The binary version includes its own dependencies.
- `python-dotenv`: Used to load environment variables from a `.env` file (for database credentials).

### 7. Create a `.env` File for Database Credentials
The script `db_infer_listen.py` expects the PostgreSQL password to be stored in an environment variable `PG_PASS`. You can provide this by creating a `.env` file in the same directory as the script.

Create a file named `.env` with the following content:
```env
PG_PASS='your_actual_database_password'
```
Replace `your_actual_database_password` with your actual PostgreSQL password. **Ensure this file is kept secure and not committed to version control if it contains sensitive information.** You might want to add `.env` to your `.gitignore` file.

### 8. Ensure PostgreSQL is Running and Accessible
Make sure your PostgreSQL server is:
- Running.
- Accessible from the Ubuntu server where you are running the script (e.g., networking, firewall rules).
- Configured with a database named `base` (or modify the `db_name` variable in the script).
- Has a user `postgres` with the password specified in `PG_PASS` (or modify `db_user` in the script).
- Has the table `server_inference` with a `server_infer_id` column.

### 9. Run the Script
You can now run the Python script:
```bash
python db_infer_listen.py
```
The script contains an example usage in its `if __name__ == "__main__":` block, which you can modify to test with specific `server_infer_id`s.

### 10. Deactivate the Virtual Environment (When Done)
Once you are finished working, you can deactivate the virtual environment:
```bash
deactivate
```

This README should provide a good starting point for setting up and running your script. 