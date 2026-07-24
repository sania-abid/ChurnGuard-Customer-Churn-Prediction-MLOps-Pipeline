# This script sets up and runs the project locally without Docker.

import os
import subprocess

def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

def run_api():
    """Run the FastAPI application."""
    print("Starting the API...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "src")) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen([
        "python", "-m", "uvicorn", "churn_app.api:app", "--host", "0.0.0.0", "--port", "8000"
    ], env=env)

def run_ui():
    """Run the UI application."""
    print("Starting the UI...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "src")) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen([
        "python", "-m", "churn_app.ui"
    ], env=env)

def run_mlflow():
    """Run the MLflow server."""
    print("Starting MLflow...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "src")) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen([
        "python", "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000",
        "--backend-store-uri", "file:./mlruns/backend",
        "--default-artifact-root", "./mlruns/artifacts"
    ], env=env)

def main():
    install_dependencies()
    run_api()
    run_ui()
    run_mlflow()

if __name__ == "__main__":
    main()