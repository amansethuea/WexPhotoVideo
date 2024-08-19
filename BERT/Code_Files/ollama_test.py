import ollama
import time
import os
import sys


def fetch_reason():
            # Upgrading ollama to latest first
            os.system(f"{sys.executable} -m pip install --upgrade ollama")

            start_time = time.time()
            review = "this was not the first pre-owned item that i have purchased from wex and i would not hesitate to use them again. the description is always very accurate and priced well. i also traded in a lens that i have not used for a while. . was offered more than i expected and used this to purchase a camera body. so i would advise you all, if its not used sell it. must also add both transaction where very quick and trouble free"
            response = ollama.chat(model='llama3', messages=[{
                'role': 'user', 
                'content': f"Could you tell me if there is any issue with this customer review: '{review}'. Keep the answer within 20 words."
            }])
            end_time = time.time()
            total_time = end_time - start_time
            print(response['message']['content'])
            print(f"The total time taken to generate a response is {total_time:.2f} seconds")
fetch_reason()