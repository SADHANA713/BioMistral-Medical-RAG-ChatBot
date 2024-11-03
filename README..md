
# BioMistral Medical RAG Chatbot
This project focuses on developing a chatbot that assists users with heart-related health questions, offering insights on symptoms, treatments, and general heart health information.
## Project Overview
- Goal: Develop a chatbot to answer user queries about health concerns, symptoms, treatments, and more.
- Purpose: Serve as a reliable and accessible resource for medical information and advice.
- Learning Context: This project was completed as part of the learning stages of the GUVI SAWIT.AI Women-Only, Gen AI Learning Challenge.

## Technologies Used
- Langchain: For building the chatbot framework.
- Hugging Face Transformers: For implementing the LLM and embedding model.
- PyPDF2: For loading and parsing PDF documents.

## Key Components
  1. Llama Library :Employed for leveraging large language models (LLMs) to generate responses.
  2. Retriever : Gathers contextually relevant information to improve the accuracy of LLM responses.
  3. Prompt Template:Combines retriever output and user queries to deliver accurate, informative responses.

### Implementation
- Retrieval Augmented Generation (RAG) Chain:
  - A Retrieval-Augmented Generation (RAG) chain integrates the retriever, LLM, and prompt template. This setup dynamically retrieves relevant information and generates responses tailored to the user's medical questions.

### Development Steps
 1. Environment Setup
    - Set up the development environment in Google Colab.
    - Install necessary Python libraries (LangChain, Sentence Transformers, Hugging Face, etc.).

2. Document Preprocessing
    - Import medical documents into the environment.
    - Extract text using PyPDF2 for PDFs and python-docx for Word files.
    - Use LangChain’s text splitter to divide the text into manageable segments for efficient retrieval.

3. Creating Embeddings and Vector Store
   - Generate embeddings for each text chunk using a pre-trained embedding model from Hugging Face.
   - Build a Chroma vector store to store and retrieve embeddings efficiently.
   - Index text chunks with their embeddings for similarity-based retrieval.
4. LLM Integration
   - Load a pre-trained language model (LLM) via the Llama library for generating     answers.
   - Design a prompt template to merge the retrieved context with user queries.
   - Construct a Retrieval-Augmented Generation (RAG) chain with LangChain’s Chain class to seamlessly integrate the retriever, LLM, and prompt template.

### Testing
The chatbot was tested with various health-related prompts to ensure reliable and accurate responses.

## Models and Data Used
- [PubMedBERT Base Embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
- [BioMistral-7B Model](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/tree/main)
- [Healthy Heart PDF by NHLBI](https://www.nhlbi.nih.gov/files/docs/public/heart/healthyheart.pdf)






