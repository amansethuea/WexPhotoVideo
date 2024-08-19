import ollama

def fetch_reason():
        # Make sure to filter out 'No Issue' rows
        #issues = row['Issues'].strip('{}').split(', ')
        #if pd.notna(row['Issues']) and row['Issues'] != '' and 'No Issue' not in issues:
            #review = row['Content']
            review = "this was not the first pre-owned item that i have purchased from wex and i would not hesitate to use them again. the description is always very accurate and priced well. i also traded in a lens that i have not used for a while. . was offered more than i expected and used this to purchase a camera body. so i would advise you all, if its not used sell it. must also add both transaction where very quick and trouble free"
            response = ollama.chat(model='llama3', messages=[{
                'role': 'user', 
                'content': f"Could you tell me if there is any issue with this customer review: '{review}'. Keep the answer within 20 words."
            }])
            print(response['message']['content'])

fetch_reason()