import os
import gradio as gr
import validators  # For URL validation
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Global visitor counter
visitor_count = 0

# Function to extract job details
def extract_job_details(job_url):
    if not validators.url(job_url):
        raise ValueError("The provided URL is not valid. Please enter a proper URL starting with http or https.")

    # LLM setup
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0,
        groq_api_key="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"  # Replace with your actual API key
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
    global visitor_count
    visitor_count += 1  # Increment the visitor counter
    try:
        job_details = extract_job_details(job_url)
        formatted_output = format_job_details(job_details)
        return f"**Visitor Count**: {visitor_count}\n\n" + formatted_output
    except Exception as e:
        return f"**Visitor Count**: {visitor_count}\n\nAn error occurred: {e}"

# Custom footer component
def footer_component():
    return """
    <div style="text-align: center; margin-top: 20px; font-size: 14px;">
        <a href="https://www.linkedin.com/in/the-sandeep-kumar" target="_blank" style="text-decoration: none; color: #0073b1;">
            Created by Sandeep Kumar 😌
        </a>
    </div>
    """

# Gradio interface setup
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter Job Posting URL", placeholder="https://example.com/job", lines=3, elem_id="large-input"),
    outputs=gr.Markdown(label="Formatted Job Details"),
    live=True
)

# Add the footer and remove "Flag" button by customizing CSS
interface = interface.queue()

# Port Configuration and Launch
port = int(os.environ.get("PORT", 10000))  # Dynamically assign port from Render's environment
interface.launch(
    server_name="0.0.0.0", 
    server_port=port, 
    share=False,  # Render expects direct port binding
    show_footer=False, 
    custom_footer=footer_component(),
    theme="default",
    css="""
        #large-input { width: 100%; height: 120px; font-size: 16px; }
        button[title="Flag"] { display: none !important; }
    """
)
