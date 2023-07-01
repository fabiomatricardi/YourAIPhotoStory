# libraries for AI inferences
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
# Internal usage
import os
import datetime
import streamlit

yourHFtoken = "hf_xxxxxxxxxx"   #paste here your HF token

# Only HuggingFace Hub Inferences
model_TextGeneration="togethercomputer/RedPajama-INCITE-Chat-3B-v1"
model_Image2Text = "Salesforce/blip-image-captioning-base"
model_Text2Image="runwayml/stable-diffusion-v1-5"
model_Summarization="MBZUAI/LaMini-Flan-T5-248M"
model_Text2Speech="espnet/kan-bayashi_ljspeech_vits"

def imageToText(url):
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=yourHFtoken)
    model_Image2Text = "Salesforce/blip-image-captioning-base"
    # tasks from huggingface.co/tasks
    text = client.image_to_text(url,
                                model=model_Image2Text)
    print(text)
    return text

basetext = imageToText("./family.jpg")

# Langchain to HuggingFace Inferences
def LC_TextGeneration(model, basetext):
    from langchain import PromptTemplate, LLMChain
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
    llm = HuggingFaceHub(repo_id=model , model_kwargs={"temperature":0.45,"min_length":30, "max_length":250})
    print(f"Running repo: {model}")    
    print("Preparing template")
    template = """<human>: write a very short story about {basetext}.
    The story must be a one paragraph.
    <bot>: """
    prompt = PromptTemplate(template=template, input_variables=["basetext"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    start = datetime.datetime.now() #not used now but useful
    print("Running chain...")
    story = llm_chain.run(basetext)
    stop = datetime.datetime.now() #not used now but useful
    elapsed = stop - start
    print(f"Executed in {elapsed}")
    print(story)
    return story


# Variable and Inference to HF with LangChain
basetext = "a family walking on the beach at sunset"


mystory = LC_TextGeneration(model_TextGeneration, basetext)
print("="*50)
finalstory = mystory.split('\n\n')[0]
print(finalstory)


def  text2speech(text):
  import requests
  API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
  headers = {"Authorization": f"Bearer {yourHFtoken}"}

  payloads = {
      "inputs" : text
  }
  response = requests.post(API_URL, headers=headers, json=payloads)
  with open('audio.flac', 'wb') as file:
    file.write(response.content)
  print("audio.flac file is created")

mytext = "The sun was setting over the horizon, casting a warm orange glow over the beach. The family was walking along the shore, enjoying the last few moments of the day."
text2speech(mytext)

