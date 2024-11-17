import gradio as gr
import validators  # For URL validation
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Function to extract job details
def extract_job_details(job_url):
    if not validators.url(job_url):
        raise ValueError("The provided URL is not valid. Please enter a proper URL starting with http or https.")

    # LLM setup
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0,
        groq_api_key="YOUR_API_KEY"  # Replace with your actual API key
    )

    # Loading webpage content
    loader = WebBaseLoader(job_url)
    pg_data = loader.load()

    if not pg_data:
        raise ValueError("No content found on the provided URL.")
    
    page_content = pg_data.pop().page_content

    # Prompt for extracting job details
    prompt_extract = PromptTemplate.from_template(
        """
        ### Scraped Text From Website
        {page_data}
        ### Instruction:
        The scraped text is from the Career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing
        the following keys: 'role', 'experience', 'skills', and 'description'.
        Ensure that the JSON is structured as follows:
        {{
            "job_postings": [
                {{
                    "role": "Job Title",
                    "experience": "Experience Level",
                    "skills": ["Detailed Skill Description 1", "Detailed Skill Description 2"],
                    "description": "Job Description"
                }}
            ]
        }}
        Please ensure that the skills section includes exact descriptions as written on the page. Do not summarize or shorten the information. Ensure that the output is valid JSON and does not include any additional text or explanation.
        """
    )

    # Extract job details using the LLM
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_content})

    # Parse the response as JSON
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)

    return json_res

# Function to format the extracted job details
def format_job_details(job_details):
    formatted_output = ""
    job_postings = job_details.get("job_postings", [])

    for index, job in enumerate(job_postings, start=1):
        formatted_output += f"### Job Posting {index}:\n\n"
        formatted_output += f"**Role**: {job.get('role', 'N/A')}\n\n"
        formatted_output += f"**Experience**: {job.get('experience', 'N/A')}\n\n"
        
        # Combine experience with skills
        skills = job.get('skills', [])
        experience = job.get('experience', 'N/A')
        
        formatted_output += "**Skills and Experience**:\n"
        
        if skills:
            for skill_index, skill in enumerate(skills, start=1):
                formatted_output += f"  - {skill}\n"
            formatted_output += f"  **Experience**: {experience}\n\n"
        else:
            formatted_output += "  - No skills listed.\n\n"
        
        formatted_output += f"**Description**: {job.get('description', 'N/A')}\n\n"
        formatted_output += "---\n"  # Add a separator between job postings

    return formatted_output.strip()  # Remove trailing whitespace

# Gradio UI function
def gradio_interface(job_url):
    try:
        job_details = extract_job_details(job_url)
        formatted_output = format_job_details(job_details)
        return formatted_output
    except Exception as e:
        return f"An error occurred: {e}"

# Gradio interface setup
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter Job Posting URL", placeholder="https://example.com/job", lines=2),
    outputs=gr.Markdown(label="Formatted Job Details"),
    live=True
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
