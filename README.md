# NLP LLM-Configurable-GuardRail
This
## Setting up the environment
### Other environments
The details set here are for .venv environments, and the .bat file reflects that. Edit the .bat file accordingly
or run it manually in your preferred environment.
### .venv
Set your virtual environment with the following code in the terminal.
```python
python -m venv venv
```
Ensure that your terminal is running on the .venv (activate.ps1 is run)
### Downloading the requirements
#### Manual Download
Make sure to download the libraries in the .venv file in the project directory
```python
pip install transformers
pip install streamlit
pip install torch #Insert your CUDA version here
pip install pyperclip
pip install accelerator
pip install ...
```
#### Using requirements.txt
You may also run requirements.txt to run the original setup
```python
pip install -r requirements.txt
```
## Running the website
## Manual Running
The python module FrontEnd.py uses streamlit, to run it, run the following code (Assuming you are in the project directory)
```python
streamlit run Scripts/FrontEnd.py
```
### Bat File (.venv only)
Once the .venv environment is setup and all libraries are installed.    
RunStreamLit.bat to open the terminal and start the localhost.
