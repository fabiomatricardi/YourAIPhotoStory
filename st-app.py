# Python app for HuggingFace Inferences
# Only API Access token from Huggingface.co
# libraries for AI inferences
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
# Internal usage
import os
import datetime
import streamlit as st


yourHFtoken = "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"
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




def Text2Image(text):
  from huggingface_hub import InferenceClient
  client = InferenceClient(model="runwayml/stable-diffusion-v1-5", token=yourHFtoken)
  image = client.text_to_image(text)
  image.save("yourimage.png")

#myiimage = Text2Image("An astronaut riding a horse on the moon.")


def  text2speech(text):
  import requests
  API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
  headers = {"Authorization": f"Bearer {yourHFtoken}"}

  payloads = {
      "inputs" : text
  }
  response = requests.post(API_URL, headers=headers, json=payloads)
  with open('audiostory.flac', 'wb') as file:
    file.write(response.content)




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





# Langchain to HuggingFace Inferences
def LC_TextGeneration(model, basetext):
    from langchain import PromptTemplate, LLMChain
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
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



def main():

  st.set_page_config(page_title="Your Photo Story Creatror App", page_icon='📱')

  st.header("Turn your Photos into Amazing Audio Stories")
  st.divider()
  st.header("🖼️ --> 🏰 --> 📻")
  
  image_file = st.file_uploader("Choose an image...", type='jpg')
  if image_file is not None:
    print(image_file)
    bytes_data = image_file.getvalue()
    with open(image_file.name, "wb") as file:
      file.write(bytes_data)
    st.image(image_file, caption="Uploaded Image...",
             use_column_width=True)
    
    st.warning("Generating Photo description",  icon="🤖")
    basetext = imageToText(image_file)
    with st.expander("Photo Description"):
      st.write(basetext)    
    st.warning("Generating Photo Story",  icon="🤖")
    mystory = LC_TextGeneration(model_TextGeneration, basetext)
    finalstory = mystory.split('\n\n')[0]
    with st.expander("Photo Story"):
      st.write(finalstory)    
    st.warning("Generating Audio Story",  icon="🤖")
    text2speech(finalstory)
    
    #with st.expander("Photo Description"):
    #  st.write(basetext)
    #with st.expander("Photo Story"):
    #  st.write(finalstory)
    
    st.audio('audiostory.flac')
    st.success("Audio Story completed!")


if __name__ == '__main__':
   main()