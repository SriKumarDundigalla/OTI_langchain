import os
import PyPDF2
import openai
import tiktoken
from dotenv import load_dotenv
import logging
import nbformat
# Load environment variables from the .env file
load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import warnings
import shutil
from langchain_text_splitters import TokenTextSplitter
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

import logging
import nbformat
import PyPDF2

def read_file_content(file_info):
    """
    Reads the content of a file based on its type and returns the content as a string.

    :param file_info: Dictionary containing the file's details.
    :return: Content of the file as a string.
    """
    file_path = file_info['path']
    file_type = file_info['type']
    content = ''

    try:
        if file_type == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text()
        elif file_type == '.ipynb':
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
                for cell in nb.cells:
                    if cell.cell_type == 'code':
                        content += cell.source + '\n\n'  # Add code cell content
                    elif cell.cell_type == 'markdown':
                        content += cell.source + '\n\n'  # Add markdown cell content
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        logging.info(f"Successfully read content from: {file_path}")
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
    return all_page_contents
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

def generate_learning_outcomes_for_chunks(documents):
    api_key = os.getenv('OPENAI_API_KEY')
    delimiter = "####"
    chunk_LOs = {}  # Dictionary to hold learning outcomes for each chunk

    # Initialize OpenAI client with your API key
    client = openai.OpenAI(api_key=api_key)

    # The number of outcomes to generate per chunk, adjust as needed or dynamically set
    number_of_outcomes = int(os.getenv('LOs_PER_CHUNK', 5))

    system_message = f"""
    As a professor tasked with developing content for a business analytics course, you will be provided with course material. \
    This material is segmented by {delimiter} for organizational purposes. \
    Your objective is to distill this material into {number_of_outcomes} distinct learning outcomes. \
    These outcomes should encapsulate the core skills and knowledge that students are expected to gain, ensuring each outcome is unique and avoids overlap.

    Please format your output as a Python list, where each item is a string representing a unique learning outcome. 

    For example:
    learning_outcomes = [
        "Understand and apply foundational principles of business analytics.",
        "Employ analytical techniques to address and solve business challenges.",
        "Assess the effectiveness of various business analytics strategies.",
        "Develop analytics solutions tailored to real-world business scenarios.",
        "Critically interpret data to make informed business decisions."
    ]
    learning_outcomes = [
    "Gain proficiency in fundamental programming concepts.",
    "Develop problem-solving skills through coding exercises.",
    "Create software applications to address real-world problems.",
    "Collaborate effectively in a team-based programming environment.",
    "Critically evaluate code to identify and fix errors."
    ]

    learning_outcomes = [
            "Master the principles of data structures and algorithms.",
            "Apply algorithms to solve complex computational problems.",
            "Analyze the efficiency of algorithms through Big O notation.",
            "Design and implement data structures for efficient data storage.",
            "Optimize algorithms for improved performance."
        ]

        learning_outcomes = [
            "Understand the theory and application of machine learning algorithms.",
            "Implement machine learning models to analyze datasets.",
            "Evaluate model performance and interpret results.",
            "Apply machine learning techniques to various domains such as finance, healthcare, etc.",
            "Critically assess the ethical implications of machine learning applications."
        ]

        learning_outcomes = [
            "Comprehend the principles of cybersecurity and network security.",
            "Identify common cybersecurity threats and vulnerabilities.",
            "Implement security measures to protect against cyber attacks.",
            "Conduct penetration testing to assess system security.",
            "Develop strategies for incident response and recovery."
        ]

        learning_outcomes = [
            "Acquire knowledge of cloud computing fundamentals.",
            "Deploy applications on cloud platforms like AWS, Azure, or Google Cloud.",
            "Optimize cloud infrastructure for scalability and cost-effectiveness.",
            "Implement cloud security measures to safeguard data.",
            "Utilize cloud services for data storage, computation, and analytics."
        ]

    

    Ensure that each learning outcome is comprehensive, addressing the essential aspects of business analytics, and remains unique to avoid redundancy.
    """

    # Generate learning outcomes for each chunk
    for index, chunk in enumerate(documents, start=1):
        user_message = f"{delimiter}I request the generation of learning outcomes from the following content:{delimiter}{chunk}{delimiter} Please ensure each outcome is unique and comprehensively covers the essential skills and knowledge expected to be gained from this material."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        print(summary)
        # Find the start of the list and extract the outcomes directly
        outcomes_start = summary.find('learning_outcomes = [') + len('learning_outcomes = [')
        outcomes_end = summary.rfind(']')
        outcomes_str = summary[outcomes_start:outcomes_end]
        LOs = [outcome.strip().strip('"').strip("'") for outcome in outcomes_str.split(',') if outcome.strip()]
        
    #     # Assigning the parsed outcomes to the corresponding chunk
    #     chunk_LOs[f'Chunk {index}'] = LOs
    # all_outcomes = []
    # for outcomes in chunk_LOs.values():
    #     all_outcomes.extend(outcomes)

        
    # system_message = f"""
    #                     {delimiter}Introduction to Delimiters{delimiter}
    #                     Delimiters are used in this instruction to separate different sections of the text for organizational purposes. They help in distinguishing instructional parts, examples, and specific tasks within the prompt.

    #                     {delimiter}Task{delimiter}
    #                     You will be provided with a list of learning outcomes that have been generated from previous analyses. From this list, your objective is to distill and consolidate the information into a final list of exactly {number_of_outcomes} learning outcomes. This refined list must:

    #                     - Eliminate thematic overlap by ensuring each learning outcome covers a unique and distinct topic, thereby avoiding redundancy.
    #                     - Capture and reflect a comprehensive understanding of the key knowledge and skills that were highlighted across the initial set of outcomes, ensuring a broad yet specific coverage of the subject matter.

    #                     {delimiter}Check the Conditions{delimiter}
    #                     - The final list must exactly contain {number_of_outcomes} learning outcomes. Recheck the lenght of list before giving response.
    #                     - Each learning outcome should be distinct, capturing unique aspects found within the original list of outcomes.

    #                     {delimiter}Output Format{delimiter}
    #                     Provide your consolidated learning outcomes in the form of a Python list named 'learning_outcomes'.The list have only {number_of_outcomes} unique Learning Outcomes . Follow the structure shown below for your output:

    #                     ```python
    #                     learning_outcomes = [
    #                             "Understand and apply foundational principles of business analytics.",
    #                             "Employ analytical techniques to address and solve business challenges.",
    #                             "Assess the effectiveness of various business analytics strategies.",
    #                             "Develop analytics solutions tailored to real-world business scenarios.",
    #                             "Critically interpret data to make informed business decisions."
    #                         ]
    #                                             learning_outcomes = [
    #                         "Analyze and visualize large datasets to extract meaningful insights.",
    #                         "Utilize machine learning algorithms to predict trends and patterns.",
    #                         "Implement data cleaning and preprocessing techniques to improve dataset quality.",
    #                         "Interpret the results of data analysis in the context of business decision-making.",
    #                         "Apply ethical considerations in the collection, analysis, and use of data."
    #                     ]

    #                                                 learning_outcomes = [
    #                             "Design, develop, and deploy scalable software applications.",
    #                             "Apply software development lifecycle models to manage project progress.",
    #                             "Incorporate software testing methodologies to ensure application reliability.",
    #                             "Utilize version control systems for collaborative software development.",
    #                             "Assess user requirements and translate them into technical specifications."
    #                         ]
    #                                                 learning_outcomes = [
    #                         "Identify and mitigate potential security threats and vulnerabilities.",
    #                         "Implement encryption algorithms to protect data integrity and confidentiality.",
    #                         "Develop policies and procedures to manage enterprise security risks.",
    #                         "Conduct digital forensics investigations to respond to security incidents.",
    #                         "Evaluate the ethical and legal implications of cybersecurity measures."
    #                     ]

    #                             learning_outcomes = [
    #                                 "Design and execute digital marketing campaigns across various platforms.",
    #                                 "Utilize data analytics tools to measure campaign performance and ROI.",
    #                                 "Employ search engine optimization techniques to enhance online visibility.",
    #                                 "Create engaging content that aligns with consumer behavior and trends.",
    #                                 "Develop a comprehensive digital marketing strategy that supports business goals."
    #                             ]

    #                            """
    # user_message = f"""{delimiter} Please review and enhance the list of learning outcomes based on the following list provided. Here is the list:".  The list:{delimiter}{all_outcomes}{delimiter}."""

    # summary = response.choices[0].message.content.strip()
    # print(summary)
    # # Find the start of the list and extract the outcomes directly
    # outcomes_start = summary.find('learning_outcomes = [') + len('learning_outcomes = [')
    # outcomes_end = summary.rfind(']')
    # outcomes_str = summary[outcomes_start:outcomes_end]
    # LOs = [outcome.strip().strip('"').strip("'") for outcome in outcomes_str.split(',') if outcome.strip()]
    # return LOs

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
        documents = process_and_insert_contents(file_contents, persist_directory)
         # Summarize the content of the files using the OpenAI API
        summarized_contents = summarize_files(file_contents)
        chunked_contents = create_chunks_from_content_greedy(summarized_contents,context_window_size)
        learning_outcomes_by_chunk = generate_learning_outcomes_for_chunks(chunked_contents)
        for chunk in learning_outcomes_by_chunk:
            logging.info(f"{chunk}'\n'")


    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")