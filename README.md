# Semantic Search Web App (Chat with PDF) with Streamlit and Azure OpenAI Service


In this repository, you will discover how [Streamlit](https://streamlit.io/), a Python framework for developing interactive data applications, can work seamlessly with [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)'s [Embedding](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#embeddings-models) and [GPT 3.5](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/legacy-models#gpt-35-models) models. These tools make it possible to create a user-friendly web application that enables users to ask questions in natural language about a PDF file they have uploaded. It is a simple yet effective solution that allows users to retrieve valuable information from the document by semantic searching.

* [app.py](https://github.com/easonlai/chat_with_pdf_streamlit/blob/main/app.py) <-- Sample using FAISS (Facebook AI Similarity Search) as a Vector Database to store the embedding vectors and perform similar searches.

Architecture
![alt text](https://github.com/easonlai/chat_with_pdf_streamlit/blob/main/git-images/git-image-1.png)

To run this Streamlit web app
```
streamlit run app.py
```

Enjoy!
