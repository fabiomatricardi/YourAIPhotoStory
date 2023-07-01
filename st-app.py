# Python app for HuggingFace Inferences  with Streamlit
# libraries for AI inferences
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
# Internal usage
import os
import datetime
# STREAMLIT
import streamlit as st


yourHFtoken = "hf_xxxxxxx"  #your HF token here
# Only HuggingFace Hub Inferences

model_TextGeneration="togethercomputer/RedPajama-INCITE-Chat-3B-v1"
model_Image2Text = "Salesforce/blip-image-captioning-base"
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



def main():

  st.set_page_config(page_title="Your Photo Story Creatror App", page_icon='📱')

  st.header("Turn your Photos into Amazing Audio Stories")
  st.image('banner.png', use_column_width=True)
  st.markdown("1. Select a photo from your pc\n 2. AI detect the photo description\n3. AI write a story about the photo\n4. AI generate an audio file of the story")
  
  # test with Image by <a href="https://pixabay.com/users/michelle_maria-165491/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=7091934">Michelle Raponi</a> from <a href="https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=7091934">Pixabay</a>
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
    

    st.audio('audiostory.flac')
    st.success("Audio Story completed!")


if __name__ == '__main__':
   main()