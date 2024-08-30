# data2day-RAG-Demo

This repository contains the code and data behind the RAG and Observability demo by inovex at data2day 2024.

## Get started

### Install poetry

```bash
python3 -m venv name_of_virtualenv
source name_of_virtualenv/bin/activate
pip install poetry
```
### Install requirements

Install the dependencies for this project with poetry

```bash
poetry install
```

### Set environment variables
Store the necessary information about your Azure OpenAI model deployments in the _.env_ file.

### Launch the app
To start the chat app execute

```bash
chainlit run app.py
```

This will start the chainlit server with the chat interface at http://localhost:8000
as well as the Phoenix server for tracking and evaluating the chat interactions at http://localhost:6006/.
If you need more information on how Phoenix traces the conversation and generates evaluations
you can find the documentation right [here](https://docs.arize.com/phoenix).

## Disclaimer

The code is currently based on using the Azure OpenAI Service through LangChain.
However, it is easily adaptable to work with other model providers by simply exchanging
the model constructors in the files _app.py_ and _vectorstore.py_ with those appropriate for your models of choice.
For more information check out [this site](https://js.langchain.com/v0.2/docs/integrations/chat/) in the LangChain documentation.