import ollama
import time
import os
import sys


def fetch_reason():
    # Upgrading ollama to latest first
    os.system(f"{sys.executable} -m pip install --upgrade ollama")
    start_time = time.time()
    #review = "this was not the first pre-owned item that i have purchased from wex and i would not hesitate to use them again. the description is always very accurate and priced well. i also traded in a lens that i have not used for a while. . was offered more than i expected and used this to purchase a camera body. so i would advise you all, if its not used sell it. must also add both transaction where very quick and trouble free"
    
    '''
    review = """
        have just wasted 90 minutes driving to and from moor allerton as i needed a fast sony lens for a wedding tomorrow. There were two staff in the branch and one other customer when i went in. for some bizarre
reason both staff members saw fit to ignore me and both dealt with the one customer. one said he would be with me in a minute and went out back to find something for the one customer they had,
he was actually gone about 5 or 10 minutes, came back with something like a lens cap, then they both spent even more time with the same customer. the same staff member then walks past me again on his
way to the store and says, i will be with you in a minute, you know, like he would said 10 minutes before. by the his point i would been stood by the counter for 15- 20 mins and i told him not to bother as I was leaving. so wex lost a decent sale and I will never shop there again. appalling and rude customer service!
    """
    '''


    '''    
    review = """
    initial offer was fine but reduced by around 10% on a boxed lens in good condition. took quite a few days to make the offer after the lens was received. however the money owed was paid promptly. i do not
plan to sell anything else through wex

    """
    '''

    
    review = """
        love my new camera, got a nikon zf from here and the service in-store and online was really good. belfast branch let me see and use the display model they had, and answered all my questions in a helpful and
friendly way, the also pointed out key features and things to consider plus alternatives if needed.

        """
    

    response = ollama.chat(model='llama3', messages=[{
          'role': 'user', 
          'content': f"Could you tell me if there is any issue with this customer review: '{review}'. Keep the answer within 10 words and as comma separated keywords. The keywords must contain if there is any electronic product mentioned only if there is any issue detected in the review"
          }])
    end_time = time.time()
    total_time = end_time - start_time
    print(response['message']['content'])
    print(f"The total time taken to generate a response is {total_time:.2f} seconds")
fetch_reason()