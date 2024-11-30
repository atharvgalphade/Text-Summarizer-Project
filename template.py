import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

project_name="TextSummarizer"

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"
]

for file_path in list_of_files:
    file_path=Path(file_path)  #This will give me the path of file according to linux and windows
    filedir, filename= os.path.split(file_path) #This will split the path into its components
    
    if filedir!="":
        os.makedirs(filedir, exist_ok=True)  #This will create the directory if it doesn't exist
        logging.info(f"Creating directory:{filedir} for the file {filename}")
        
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,'w') as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
            
            
            
    else:
        logging.info(f"{file_path} already exists")