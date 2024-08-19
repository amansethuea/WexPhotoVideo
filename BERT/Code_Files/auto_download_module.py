import os
import sys

"""
NOTE: Run only after activing virtual env
Step 1 (If venv not already created): python -m venv myenv (myenv is name of virtual environment)
Step 2: myenv\Scripts\activate (on windows machine)
"""


class ModuleDownload(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.requirements_txt_file = os.getcwd() + "/../Required_Modules/requirements.txt"
        elif sys.platform.startswith("darwin"):
            self.requirements_txt_file = os.getcwd() + "/Required_Modules/requirements.txt"
        elif sys.platform.startswith("linux"):
            self.requirements_txt_file = os.getcwd() + "/../Required_Modules/requirements.txt"

    def auto_module_download(self):
        print("Installing packages in requirement.txt")
        os.system(f"{sys.executable} -m pip install -r {self.requirements_txt_file}")
        print("Packages Installed. Proceeding")
        print("Upgrading Ollama package explicitly to latest irrespective of mentioned version in requirements.txt")
        os.system(f"{sys.executable} -m pip install --upgrade ollama")
        print("Downloading spaCy large pre-trained model")
        os.system(f"{sys.executable} -m spacy download en_core_web_lg")
        print()


if __name__ == "__main__":
    obj = ModuleDownload()
    obj.auto_module_download()