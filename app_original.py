# libraries for AI inferences
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
# Internal usage
import os
import datetime
import streamlit

yourHFtoken = "hf_xxxxxx"

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

basetext = imageToText("./photo.jpeg")


def Text2Image(text):
  from huggingface_hub import InferenceClient
  client = InferenceClient(model="runwayml/stable-diffusion-v1-5", token=yourHFtoken)
  image = client.text_to_image(text)
  image.save("yourimage.png")

myiimage = Text2Image("An astronaut riding a horse on the moon.")


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

mytext = "So let's create a function for our text-to-speech generation with the requests method"
text2speech(mytext)


def generation(question):
  from langchain import HuggingFaceHub
  os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
  repo_id = "MBZUAI/LaMini-Flan-T5-248M"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
  llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 64})
  from langchain import PromptTemplate, LLMChain
  template = """Question: {question}

  Answer: Let's think step by step."""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  print(llm_chain.run(question))

question = "Who won the FIFA World Cup in the year 1994? "
generation(question)


def summary(text):
  os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
  from langchain import HuggingFaceHub
  from langchain.chains.summarize import load_summarize_chain
  llm = HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-248M", model_kwargs={"temperature":0, "max_length":512})
  from langchain.document_loaders import TextLoader
  from langchain.docstore.document import Document
  docs = [Document(page_content=text)]
  chain = load_summarize_chain(llm, chain_type="map_reduce")
  summary = chain.run(docs)
  return summary


mystrangeText = """What I am going to tell you now is a real tip: you cannot do all the inferences with the API token alone.
I know, looks like I messed up with you… But I didn't. There is a way for everything and I have you covered (for what you usually cannot do) in the next section.
So Hugging Face API inference and its Transformers library are our gateway to all the Large Language Models."""
summary(mystrangeText)




# Langchain to HuggingFace Inferences
def LC_TextGeneration(model, basetext):
    from langchain import PromptTemplate, LLMChain
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxx"
    #llm = HuggingFaceHub(repo_id=model , model_kwargs={"temperature":0.45,"min_length":40, "max_length":130})
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
basetext = "a couple walking on the beach"
secondtext = "A man and a dog"


mystory = LC_TextGeneration(model_TextGeneration, basetext)
print("="*50)
finalstory = mystory.split('\n\n')[0]
print(finalstory)


#togethercomputer/RedPajama-INCITE-Instruct-3B-v1

"""
template = <human>: write a short story in two paraghraphs about {basetext}.
    <bot>:
Paragraph 1:

The beach was empty.

The couple walked along the shore, hand in hand.

The sun was shining, and the sky was blue.

The sand was warm under their feet.

The waves lapped gently against the shore.

The couple smiled at each other.

Paragraph 2:

They walked for hours,
talking about their lives and their hopes and dreams.

The sky began to turn dark,
###############three paraghraphs #############
Paragraph 1:

The beach was empty.

The couple walked along the sand,
holding hands,
looking out at the ocean.

Paragraph 2:

The sun was setting,
casting a warm glow
on the water and the sky.

The couple sat down on a rock,
and leaned against each other,
looking out at the horizon.

Paragraph 3:

The sky turned pink,
and the sun dipped below the horizon
"""

