from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

def generate(query, docs):
    """Generate answer using advanced RAG chain with prompt template."""
    # Initialize LLM with configuration
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        top_p=config.LLM_TOP_P
    )
    
    # Create prompt template
    prompt = PromptTemplate.from_template(config.LLM_PROMPT_TEMPLATE)
    
    # Format context from retrieved documents
    context = "\n\n".join([d.page_content for d in docs])
    
    # Create input dict for the chain
    chain_input = {"context": context, "question": query}
    
    # Create and invoke RAG chain
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke(chain_input)
    
    return response
