import os
from time import sleep

import pandas as pd
from dotenv import load_dotenv

import chainlit as cl
import phoenix as px
from langchain.chains import create_retrieval_chain

# add imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from phoenix.evals import (
    HallucinationEvaluator,
    LiteLLMModel,
    run_evals,
    ToxicityEvaluator,
    RelevanceEvaluator,
    QAEvaluator,
)
from phoenix.trace import SpanEvaluations
from phoenix.trace.langchain import LangChainInstrumentor
from vectorstore import create_vectorstore
from utils import insert_span_ids

load_dotenv("../.env")


async def initialize_chain():
    # initialize the Azure OpenAI Model
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        openai_api_type="azure",
        temperature=0.0,
        streaming=True,
    )

    # creates vectorstore if it does not exist already and load data
    vectorstore = await create_vectorstore(
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        embeddings_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )

    # initialize a retriever from the vectorstore
    retriever = vectorstore.as_retriever()

    # system prompt to create history or rather add context
    contextualize_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, 
        formulate a standalone question which can be understood 
        without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""

    # prompt template for context + placeholder for chat history
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_prompt
    )

    # crate a system prompt that tells the LLM to answer questions based on the given context
    # and use a variable that represents the context
    system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer 
        the question. If you don't know the answer, say that you 
        don't know. Use three sentences maximum and keep the 
        answer concise.
        \n\n
        {context}"""

    # prompt template with system prompt + placeholder for chat history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # create a helper chain that inserts the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(model, prompt)

    # create the final RAG chain
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return chain


async def initialize_evaluators():
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_VERSION")

    evaluator_model = LiteLLMModel(
        model=f'azure/{os.getenv("AZURE_OPENAI_DEPLOYMENT")}'
    )

    toxicity_evaluator = ToxicityEvaluator(evaluator_model)
    hallucination_evaluator = HallucinationEvaluator(evaluator_model)
    relevance_evaluator = RelevanceEvaluator(evaluator_model)
    qa_evaluator = QAEvaluator(evaluator_model)

    return (
        toxicity_evaluator,
        hallucination_evaluator,
        relevance_evaluator,
        qa_evaluator,
    )


@cl.on_chat_start
async def on_chat_start():
    # start a phoenix session
    session = px.launch_app()

    # initialize Langchain auto-instrumentation
    LangChainInstrumentor().instrument()

    # create chain and save it in user session
    chain = await initialize_chain()
    cl.user_session.set("chain", chain)

    # init chat history
    cl.user_session.set("chat_history", [])

    # initialize Phoenix evaluator and save it in user session
    toxicity_evaluator, hallucination_evaluator, relevance_evaluator, qa_evaluator = (
        await initialize_evaluators()
    )
    cl.user_session.set("toxicity_evaluator", toxicity_evaluator)
    cl.user_session.set("hallucination_evaluator", hallucination_evaluator)
    cl.user_session.set("relevance_evaluator", relevance_evaluator)
    cl.user_session.set("qa_evaluator", qa_evaluator)


@cl.on_message
async def on_message(message: cl.Message):
    # get initialized chain from user session
    chain = cl.user_session.get("chain")

    # get chat history from user session
    chat_history = cl.user_session.get("chat_history")

    # placeholder for model answer
    answer_message = cl.Message(content="", author="Chatbot")

    # update chain with user message
    answer = ""
    async for chunk in chain.astream(
        {"input": message.content, "chat_history": chat_history},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        answer += chunk

    # return answer to user
    answer_message.content = answer["answer"]
    await answer_message.send()

    # add question and answer to chat history
    chat_history.extend(
        [
            HumanMessage(content=message.content),
            AIMessage(content=answer["answer"]),
        ]
    )

    # get evaluators from user session
    toxicity_evaluator = cl.user_session.get("toxicity_evaluator")
    hallucination_evaluator = cl.user_session.get("hallucination_evaluator")
    relevance_evaluator = cl.user_session.get("relevance_evaluator")
    qa_evaluator = cl.user_session.get("qa_evaluator")

    eval_dict = {
        "output": [message.content],
        "input": [answer["answer"]],
        "reference": [answer["context"]],
    }
    eval_df = pd.DataFrame.from_dict(eval_dict)

    # wait a little bit in case span information hasn't become completely available yet
    sleep(3)

    # retrieve information about spans
    spans_dataframe = px.Client().get_spans_dataframe()

    # execute evaluation
    toxicity_eval_df, hallucination_eval_df, relevance_eval_df, qa_eval_df = run_evals(
        dataframe=eval_df,
        evaluators=[
            toxicity_evaluator,
            hallucination_evaluator,
            relevance_evaluator,
            qa_evaluator,
        ],
        provide_explanation=True,
    )

    # extract span ids for logging and displaying evaluations in phoenix
    toxicity_eval_df = insert_span_ids(toxicity_eval_df, spans_dataframe)
    print(toxicity_eval_df["explanation"][0])
    hallucination_eval_df = insert_span_ids(hallucination_eval_df, spans_dataframe)
    print(hallucination_eval_df["explanation"][0])
    relevance_eval_df = insert_span_ids(relevance_eval_df, spans_dataframe)
    print(relevance_eval_df["explanation"][0])
    qa_eval_df = insert_span_ids(qa_eval_df, spans_dataframe)
    print(qa_eval_df["explanation"][0])

    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
        SpanEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
        SpanEvaluations(eval_name="QA Correctness", dataframe=qa_eval_df),
        SpanEvaluations(eval_name="Toxicity", dataframe=toxicity_eval_df),
    )
