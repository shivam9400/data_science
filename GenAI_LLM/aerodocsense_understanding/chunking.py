from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pymupdf4llm
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker

def recursive_char_text_splitting(chunk_size, chunk_overlap, sep):
    ''' Character-Based Chunking '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=sep 
    )

    return text_splitter

def markdown_header_text_splitting(sep):
    ''' Structural Chunking '''

    # Define Headers to Split On (The header text becomes the metadata key)
    ### ATA 29 only has two levels of header
    headers_to_split_on = sep

    markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    # strip_headers=True is the default, ensuring the chunk only has the content.
    )

    return markdown_splitter

def token_text_splitting(chunk_size, overlap):
    ''' Fixed Size Chunking '''

    token_splitter = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size, 
        chunk_overlap=overlap
        )

    return token_splitter

def documents_to_dataframe(documents):
    data = []
    
    for doc in documents:
        # Start the dictionary with the core content
        row = {
            'Content': doc.page_content,
            'Source': doc.metadata.get('source', 'N/A')
        }
        
        # Merge all metadata keys into the row
        row.update(doc.metadata)
        data.append(row)
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 1. Load the PDF
    pdf_path = "GenAI_LLM\\aerodocsense_understanding\\data\\ata29_doc.pdf"
    loader = PyPDFLoader(pdf_path)

    ### Load the documents. This returns a list of Document objects, one for each page.
    pages = loader.load()

    ############################# FIXED SIZE CHUNKING ##################################
    fsc = token_text_splitting(chunk_size=1024, overlap=128)
    chunks = fsc.split_documents(pages)
    print(f"Total number of FIXED SIZE chunks created: {len(chunks)}")

    ############################# RECURSIVE CHARACTER SPLITTING ##################################
    # try recursive_char_text_splitting
    rcts = recursive_char_text_splitting(1000, 200, ["\n\n", "\n", " ", ""] )
    # Apply the Split
    chunks = rcts.split_documents(pages)
    print(f"Total number of RECURSIVE SPLIT chunks created: {len(chunks)}")

    ############################# MARK DOWN SPLITTING ##################################
    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    # Create a Single LangChain Document from the Markdown Text
    markdown_doc = Document(
        page_content=markdown_text,
        metadata={"source": pdf_path, "format": "markdown"} 
        )
    # try markdown_header_text_splitting
    mhts = markdown_header_text_splitting(
        [
        ("#", "Chapter"),       # Level 1 header
        ("##", "Section"),      # Level 2 header
        #("###", "Procedure")    # Level 3 header
        ]
    )
    chunks = mhts.split_text(markdown_doc.page_content)
    chunks_df = documents_to_dataframe(chunks)
    # chunks_df.to_csv("markdown_splitter.csv")
    print(f"Total number of MARKDOWN SPLIT chunks created: {len(chunks)}")

    ########################## SEMANTIC CHUNKING ###########################################
    # This model will be used by the SemanticChunker to calculate semantic distances.
    # Initialize the Embedding Model ('all-MiniLM-L12-V2')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )
    # Initialize the Semantic Chunker
    ### This means the chunker will look at semantic similarity between sentences 
    ### and decide breakpoints based on a percentile threshold
    text_splitter = SemanticChunker(
        embeddings=embeddings, 
        breakpoint_threshold_type="percentile"
    )

    # Apply the Split
    semantic_chunks = text_splitter.split_documents(pages)
    print(f"Total SEMANTIC chunks created: {len(semantic_chunks)}")

    chunk_data = []
    for i, chunk in enumerate(semantic_chunks):
        chunk_data.append({
            "chunk_id": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })
    pd.DataFrame(chunk_data).to_csv("semantic_chunks.csv", index=False)