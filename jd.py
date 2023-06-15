import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = ""

st.title('🦜🔗 Job Descriptor')
prompt = st.text_input('Enter in your prompt here') 

JD_template = PromptTemplate(
    input_variables = ['job_title'], 
    template='Generate the job description for {job_title} in 5 sentences'
)
TSK_template = PromptTemplate(
    input_variables = ['job_title'], 
    template='List the main 5 technical skills required for the {job_title} without description of those technical skills '
)
Sal_template = PromptTemplate(
    input_variables = ['job_title'], 
    template='According to payscale.com what is the average salary range for the {job_title} just the number in rupees'
)
Questions_template = PromptTemplate(
    input_variables = ['technical_skills'], 
    template='Generate three frequently asked questions with four options all being correct and ranked in percentage format 100, 75, 50 and 25 as MCQs in an interview for each of the technical skills in {technical_skills} '
)


llm = OpenAI(temperature=0.9,model_name='gpt-3.5-turbo') 
title_chain = LLMChain(llm=llm, prompt=JD_template, verbose=True, output_key='JD')
script_chain = LLMChain(llm=llm, prompt=TSK_template, verbose=True, output_key='technical_skills')
TSK_chain = LLMChain(llm=llm, prompt=Sal_template, verbose=True, output_key='Salary')
QA_chain = LLMChain(llm=llm, prompt=Questions_template, verbose=True, output_key='Questions')

sequence_chain = SequentialChain(chains=[title_chain,script_chain,TSK_chain,QA_chain],input_variables=['job_title'], output_variables=['JD','Salary','technical_skills','Questions'],verbose=True)

if prompt: 
    response = sequence_chain({'job_title':prompt })

    st.write(response) 
