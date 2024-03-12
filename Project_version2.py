import os
import PyPDF2
import openai
import tiktoken
from dotenv import load_dotenv
import logging
import nbformat
import jsonpatch
# Load environment variables from the .env file
load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import warnings
import shutil
import json
from langchain_text_splitters import TokenTextSplitter
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import nbformat
import PyPDF2
import ast
warnings.filterwarnings("ignore")
# Configure basic logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_directory(directory):
    """
    Analyzes the content of the given directory and returns details of specific file types.

    :param directory: The path of the directory to analyze.
    :return: A list of dictionaries, each containing 'name', 'type', 'size', and 'path' of the file.
    """
    logging.info(f"Analyzing directory: {directory}")
    supported_extensions = {'.md', '.ipynb','.pdf'} #'.py'
    file_details = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            extension = os.path.splitext(file_path)[1]

            if extension in supported_extensions:
                file_info = {
                    'name': file,
                    'type': extension,
                    'size': os.path.getsize(file_path),
                    'path': file_path
                }
                file_details.append(file_info)
                logging.info(f"File added for processing: {file_path}")

    return file_details


def clean_content(content):
    """
    Performs cleaning of the file content, including trimming whitespace and removing non-printable characters.
    :param content: Raw content string to be cleaned.
    :return: Cleaned content string.
    """
    content = content.strip()  # Remove leading and trailing whitespace
    content = content.replace('\x00', '')  # Remove null bytes if present
    # Normalize line breaks and whitespace
    content = content.replace('\n', ' ')  # Replace new lines with spaces
    content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with a single space

    # Remove non-printable characters
    content = ''.join(char for char in content if char.isprintable() or char in ('\n', '\t', ' '))
    # Remove non-printable characters, including the replacement character
    content = re.sub(r'[^\x20-\x7E]+', '', content)
    return content

def read_file_content(file_info):
    """
    Reads the content of a file based on its type and returns the cleaned content as a string.
    :param file_info: Dictionary containing the file's details.
    :return: Cleaned content of the file as a string.
    """
    file_path = file_info['path']
    file_type = file_info['type']
    content = ''

    try:
        if file_type == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() if page.extract_text() else ''
                    content += clean_content(page_text)
        elif file_type == '.ipynb':
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
                for cell in nb.cells:
                    cell_content = cell.source + '\n\n'  # Add cell content
                    content += clean_content(cell_content)
        else:  # Assuming '.py' or other plaintext files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = clean_content(f.read())

        logging.info(f"Successfully read and cleaned content from: {file_path}")
    except Exception as e:
        logging.exception(f"Error reading {file_path}: {e}")

    return content


def get_file_contents(file_details):
    """
    Retrieves the contents of each file based on the provided file details.

    :param file_details: List of dictionaries containing file details.
    :return: A list of dictionaries, each containing 'path' and 'content' of the file.
    """
    content_details = []
    for file_info in file_details:
        file_content = read_file_content(file_info)
        if file_content:
            content_details.append({
                'path': file_info['path'],
                'content': file_content
            })

    return content_details
def process_and_insert_contents(file_contents, persist_directory):
    """
    Processes the contents of each file, splits them, embeds, and inserts into a database.

    :param file_contents: List of dictionaries containing file paths and their contents.
    :param persist_directory: The directory to persist any necessary data for database insertion.
    """
    # Initialize the text splitter and embedding tools
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    embedding = OpenAIEmbeddings()
    all_page_contents = []  # Collect all page contents for further processing or analysis
    # Extract page_content from each Document
    for content_detail in file_contents:
        # Split the content
        documents  = text_splitter.create_documents([content_detail['content']])
        for document in documents:
            page_content = document.page_content  # Accessing the page_content attribute
            all_page_contents.append(page_content)
        # Here, you would generate embeddings and insert them into your database
        # This is a placeholder to illustrate the process
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        
        # Logging or any other operation after insertion
        logging.info(f"Processed and inserted content from: {content_detail['path']}")
    return vectordb
def summarize_files(file_details):
    """
    Processes the content of files whose content exceeds a specified global token size,
    by splitting the content into chunks. Each chunk's size is determined to ensure it 
    doesn't exceed the global token size limit. The function returns a list of dictionaries 
    with the filename/path, chunked content, and the token size of each chunk.

    :param file_details: List of dictionaries with file details.
    :return: A list of dictionaries with filename/path and chunked content.
    """
    global_token_size = int(os.getenv('GLOBAL_TOKEN_SIZE'))
    Overlap = 500  # Example overlap size, adjust as needed
    summarized_files = []

    for file in file_details:
        original_token_count = len(tiktoken.encoding_for_model("gpt-3.5-turbo-1106").encode(file['content']))

        if original_token_count > global_token_size:
            # Calculate the number of chunks needed
            N = 1 + (original_token_count - global_token_size) // (global_token_size - Overlap)

            # Initialize the splitter with calculated chunk size and overlap
            splitter = RecursiveCharacterTextSplitter( chunk_size = original_token_count // N ,  chunk_overlap = Overlap)

            # Split the content into documents/chunks
            documents = splitter.create_documents([file['content']])

            # Process each document/chunk
            for document in documents:
                page_content = document.page_content  # Assuming this is how you access the content of each chunk
                
                summarized_files.append({
                    'path': file['path'],
                    'content': page_content,
                    # Update token_count with the actual count for this chunk if necessary
                    'token_size': len(tiktoken.encoding_for_model("gpt-3.5-turbo-1106").encode(page_content))
                })
        else:
            # If the content does not exceed global token size, add it directly
            summarized_files.append({
                'path': file['path'],
                'content': file['content'],
                'token_size': original_token_count
            })

    return summarized_files

def create_chunks_from_content_greedy(file_contents, context_window_size):
    """
    Creates content chunks from a list of file content dictionaries using a Greedy approach, 
    ensuring that each chunk does not exceed a specified context window size in terms of tokens.

    Parameters:
    - file_contents (list of dict): A list of dictionaries, where each dictionary contains 
      'content' (str) and 'token_size' (int) keys. 'content' is the text of the file, and 
      'token_size' is the number of tokens that text consists of.
    - context_window_size (int): The maximum number of tokens that a single chunk can contain. 
      It defines the upper limit for the size of each content chunk.

    Returns:
    - list of str: A list of content chunks, where each chunk is a string composed of file contents 
      that together do not exceed the context window size.
    """
    all_chunks = []  # Initialize the list to hold all content chunks
    current_chunk = ""  # Initialize the current chunk as an empty string
    current_token_count = 0  # Initialize the current token count to 0

    # Sort file_contents by 'token_size' in descending order
    sorted_file_contents = sorted(file_contents, key=lambda x: x['token_size'], reverse=True)

    for content in sorted_file_contents:
        # If adding this content exceeds the context window size, start a new chunk
        if current_token_count + content['token_size'] > context_window_size:
            if current_chunk:  # Ensure the current chunk is not empty
                all_chunks.append(current_chunk)  # Add the current chunk to all_chunks
                current_chunk = ""  # Reset the current chunk
                current_token_count = 0  # Reset the token count for the new chunk

        # Add the content to the current chunk if it fits
        if current_token_count + content['token_size'] <= context_window_size:
            current_chunk += content['content'] + "\n"  # Append content and a newline for readability
            current_token_count += content['token_size']
    
    # Add the last chunk if it contains any content
    if current_chunk:
        all_chunks.append(current_chunk)

    return all_chunks
# class Learning_Outcomes(BaseModel):
#     learning_outcomes : List[str] = Field(description="list of learning outcomes")
import matplotlib.pyplot as plt
import networkx as nx

# Function to draw a graph based on given indices and title
def draw_similarity_graph(indices, cosine_sim, title,threshold):
    G = nx.Graph()

    # Add nodes and edges based on the provided indices and similarity matrix
    for i in indices:
        G.add_node(i, label=f"Doc {i+1}")
    for i in indices:
        for j in indices:
            if i != j and cosine_sim[i][j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i][j])
    
    pos = nx.spring_layout(G)  # Node positions

    plt.figure(figsize=(10, 8))
    plt.grid(True)  # Enable grid
    plt.axis('on')  # Show axis

    # Nodes
    node_colors = range(len(G))
    nx.draw_networkx_nodes(G, pos, node_size=700, cmap=plt.cm.viridis, node_color=node_colors, alpha=0.8)

    # Edges
    edges = G.edges(data=True)
    weights = [d['weight']*10 for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    plt.title(title, fontsize=20)
    plt.show()

 # Define your desired data structure for learning outcomes.
class LearningOutcomes(BaseModel):
    outcomes: List[str] = Field(description="List of learning outcomes")
 # Set up a parser to enforce the output structure.
parser = PydanticOutputParser(pydantic_object=LearningOutcomes)

def generate_learning_outcomes_for_chunks(documents):
    api_key = os.getenv('OPENAI_API_KEY')
    delimiter = "###"
    chunk_LOs = {}  # Dictionary to hold learning outcomes for each chunk

    # Initialize OpenAI client with your API key
    client = openai.OpenAI(api_key=api_key)

    # The number of outcomes to generate per chunk, adjust as needed or dynamically set
    number_of_outcomes = int(os.getenv('LOs_PER_CHUNK', 6))

    review_instructions = """
        Take time to review each learning outcome. Ensure that they are:
        - Review clarity, measurability, and alignment with the course's overarching goals. Adjust as necessary to ensure they comprehensively cover the desired skills and knowledge and that each outcome is unique.
        - Check the format of reponse to ensure they are presented as a list, with each item being a string that represents a unique learning outcome. This structured approach ensures that the learning objectives are clear, measurable, and aligned with both the course content and the educational standards set forth by Bloom's Taxonomy.
        """
    system_message = f"""

    - As an educator tasked with developing course content, you will start with an overview of the course material. This material encompasses various topics and objectives that are pivotal to the subject. Your mission is to craft {number_of_outcomes} distinct learning outcomes that students should achieve by the end of the course. These outcomes should span across different cognitive levels of Bloom's Taxonomy, ensuring a comprehensive learning journey from foundational knowledge to advanced application and innovation.
    - Please format your output as a list. 
    - To create your learning outcomes, follow these guidelines:
        1. Identify the course's main topics, themes, and objectives for context.
        2. Decide on the desired number of learning outcomes, balancing depth and manageability.
        3. Utilize Bloom's Taxonomy to select cognitive levels relevant to your course.
        4. Choose clear action verbs for each cognitive level to define expected achievements.
        5. Detail the specific knowledge and skills to be gained at each level, ensuring course content relevance.
        6. Draft your learning outcomes, incorporating the chosen action verbs and detailing the knowledge and skills.

    - Examples of Generating Learning Outcomes Lists Based on Course Content:

            question1: I request the generation of learning outcomes from the following content: Introduction to Python Programming. Please ensure each outcome is unique and comprehensively covers the essential skills and knowledge expected to be gained from this material."
            answer1 : [
                "Recall the syntax and basic data types in Python.",
                "Explain how control flow constructs work in Python.",
                "Apply functions and modules to solve a problem.",
                "Analyze a given Python code to identify inefficiencies.",
                "Evaluate different Python libraries for data analysis.",
                "Design a simple Python application that utilizes object-oriented programming concepts."
            ]

            question2 : I request the generation of learning outcomes from the following content: Basics of Environmental Science. Please ensure each outcome is unique and comprehensively covers the essential skills and knowledge expected to be gained from this material."
            answer2 : [
                "List the major components of the Earth's ecosystem.",
                "Summarize the impact of human activities on climate change.",
                "Use geographic information systems (GIS) to analyze environmental data.",
                "Differentiate between renewable and non-renewable energy sources.",
                "Judge the effectiveness of current policies on natural resource conservation.",
                "Propose a sustainable environmental management plan for a local community."
            ]

            question3 : "I request the generation of learning outcomes from the following content: Fundamentals of Economics. Please ensure each outcome is unique and comprehensively covers the essential skills and knowledge expected to be gained from this material."
            answer3 : [
                "Define key economic concepts such as supply, demand, and elasticity.",
                "Describe how markets reach equilibrium.",
                "Apply economic theories to current global economic issues.",
                "Analyze the effects of government intervention in the market.",
                "Evaluate the impact of international trade on national economies.",
                "Develop a research question related to an economic issue and outline a methodology for investigating it."
            ]

            question4 : "I request the generation of learning outcomes from the following content: Introduction to World History. Please ensure each outcome is unique and comprehensively covers the essential skills and knowledge expected to be gained from this material."
            answer4 : [
                "Identify key civilizations and their contributions to world history.",
                "Explain the causes and effects of major historical events.",
                "Compare and contrast different historical periods and their characteristics.",
                "Analyze primary and secondary sources to gain insights into historical contexts.",
                "Assess the impact of historical events on contemporary society.",
                "Create a timeline that illustrates the progression of significant historical developments."
            ]

            question5 : I request the generation of learning outcomes from the following content: Principles of Marketing. Please ensure each outcome is unique and comprehensively covers the essential skills and knowledge expected to be gained from this material."
            answer5 : [
                "List the four Ps of marketing.",
                "Interpret market research data to identify consumer behavior patterns.",
                "Apply segmentation, targeting, and positioning (STP) strategies to a marketing plan.",
                "Analyze the effectiveness of different advertising campaigns.",
                "Evaluate the role of digital marketing in modern business strategies.",
                "Design a marketing campaign for a new product or service."
            ]

        Instructions for Review and Refinement
        {review_instructions}
    """
    all_out_comes=[]
    # Generate learning outcomes for each chunk
    for index, chunk in enumerate(documents, start=1):
        user_message = f"Generate Lerning outcomes for the following content enclosed by triple hashtag{delimiter}{chunk}{delimiter}."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            max_tokens=1024,
            temperature=0
        )
        
        summary = response.choices[0].message.content

        try:
            # Attempt to parse summary as JSON in case it's a string representation of a list
            if isinstance(summary, str) and summary.startswith('[') and summary.endswith(']'):
                outcomes_list = json.loads(summary)
                
            else:
                # If summary is not a string that looks like a list, this is a fallback
                # In your case, this might be unnecessary but is kept for completeness
                outcomes_list = [summary]
            
            res = LearningOutcomes(outcomes=outcomes_list)
            all_out_comes.append(res.outcomes)
        except json.JSONDecodeError:
            print("The summary is not a valid JSON string.")
        except Exception as e:
            print(f"Error processing the response: {e}")
    print(all_out_comes)
    # Flatten each list of outcomes into a single string per list to simplify the example
    documents = [" ".join(outcome_list) for outcome_list in all_out_comes]

    # Vectorize the documents
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between documents
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Determine similarity threshold
    threshold = 0.4  # Example threshold, adjust based on your needs

    # Filter documents based on similarity (naive approach)
    filtered_indices = []
    for i in range(len(cosine_sim)):
        # If the document is not highly similar to other documents already added
        if not any(cosine_sim[i][j] > threshold and j != i for j in filtered_indices):
            filtered_indices.append(i)

    # Extract the filtered outcomes
    filtered_outcomes = [all_out_comes[i] for i in filtered_indices]

    print(f"Original number of topics: {len(all_out_comes)}")
    print(f"Filtered number of topics: {len(filtered_outcomes)}")
    # Before filtering: Use all document indices
    all_indices = range(len(cosine_sim))
    draw_similarity_graph(all_indices, cosine_sim, "Before Filtering Similarity Graph",threshold)


    draw_similarity_graph(filtered_indices, cosine_sim, "After Filtering Similarity Graph",threshold)

    return filtered_outcomes

def find_most_relevant_learning_outcome_document(vectordb, learning_outcomes):
    """
    Uses vectordb to find the most relevant learning outcome document from the database for each topic.

    :param vectordb: The vectordb instance configured for retrieval.
    :param learning_outcomes: A list of lists, where each sublist represents learning outcomes related to a specific topic.
    :return: A list of tuples, each containing the most relevant document's content and its relevance score for each list of learning outcomes.
    """
    # Initialize the vectordb retriever with 'k' set to 1 to retrieve the most relevant document
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 1})

    documents=[]
    for LOS in learning_outcomes:
        outcome_page_mapping={}
        for i in LOS:
            docs = retriever.get_relevant_documents(i)
            outcome_page_mapping[i]=docs[0].page_content
        documents.append(outcome_page_mapping)
    logging.info(documents)

def remove_old_database_files(directory_path='./docs/chroma'):
    """
    Removes the specified directory and all its contents.

    :param directory_path: Path to the directory to be removed.
    """
    try:
        # Check if the directory exists
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Remove the directory and all its contents
            shutil.rmtree(directory_path)
            logging.info(f"Successfully removed directory: {directory_path}")
        else:
            logging.info(f"Directory does not exist, no need to remove: {directory_path}")
    except Exception as e:
        logging.exception(f"Error removing directory {directory_path}: {e}")


# Main execution
if __name__ == "__main__":
    remove_old_database_files()
    # Load environment variables from the .env file
    load_dotenv()

    # Define the path of the directory to analyze
    directory_path = r"C:\Users\dsksr\Documents\BIG DATA\2024\Independent Study\QIT-LC\QTI_langchain"

    # Retrieve the OpenAI API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
     # Retrieve the OpenAI API key from environment variables
    context_window_size =int(os.getenv('context_window_size'))
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
    persist_directory = 'docs/chroma/'
    try:
        # Analyze the directory and get details of the files present
        file_details = analyze_directory(directory_path)
        # Retrieve the contents of each file from the analyzed directory
        file_contents = get_file_contents(file_details)
        # Process and insert the file contents into the database
        vectordb = process_and_insert_contents(file_contents, persist_directory)
         # Summarize the content of the files using the OpenAI API
        summarized_contents = summarize_files(file_contents)
        chunked_contents = create_chunks_from_content_greedy(summarized_contents,context_window_size)
        learning_outcomes_by_chunk = generate_learning_outcomes_for_chunks(chunked_contents)
        find_most_relevant_learning_outcome_document(vectordb,learning_outcomes_by_chunk)
        


    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")