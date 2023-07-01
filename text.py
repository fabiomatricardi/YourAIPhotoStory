from transformers import pipeline
from transformers import AutoModel
from huggingface_hub import InferenceClient
client = InferenceClient(token="hf_xxxxxxxxx")
from langchain import HuggingFaceHub
import os
import datetime

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxxxxx"
LaMini = "MBZUAI/LaMini-Flan-T5-248M" #model_TextGeneration # "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
# other models no success   openlm-research/open_llama_3b  EleutherAI/pythia-1.4b-deduped eachadea/vicuna-7b-1.1  TheBloke/vicuna-7B-1.1-HF 
# succe models  EleutherAI/pythia-2.8b  EleutherAI/gpt-neo-1.3B  togethercomputer/GPT-JT-6B-v1  
# llm = HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-248M", model_kwargs={"temperature":0.6,"min_length":150, "max_length":700})

repo="KoboldAI/GPT-J-6B-Janeway"
llm = HuggingFaceHub(repo_id=repo , model_kwargs={"temperature":0.6,"min_length":150, "max_length":500})
"""
repo="tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo , model_kwargs={"temperature":0.6,"max_length":200,
    "min_length":150, "do_sample":True,
    "top_k":10,
    "num_return_sequences":1})
"""

#KoboldAI/GPT-Neo-1.3B-Ramsay 
repo="togethercomputer/RedPajama-INCITE-Chat-3B-v1"
llm = HuggingFaceHub(repo_id=repo , model_kwargs={"temperature":0.6,"min_length":50, "max_length":150})

print(f"Running repo: {repo}")
from langchain import PromptTemplate, LLMChain
# oass template = """<|prompter|>write a long story about {basetext}.<|endoftext|><|assistant|>"""
print("Preparing template")
#template = """create a short story about {basetext}."""
template = """write a short story about {basetext}"""
basetext = "a couple walking on the beach"

template = """<human>: write a short story about {basetext}
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
