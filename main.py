import os
import gradio as gr
import validators
import json
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

# Function to clean LLM output
def clean_json_output(output):
    return output.replace("“", '"').replace("”", '"').strip()

# Function to extract and validate job details
def extract_job_details(job_url):
    if not validators.url(job_url):
        raise ValueError("The provided URL is invalid. Please use a valid URL starting with http or https.")

    # Load LLM
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0,
        groq_api_key="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"  # Replace with your actual API key
    )

    # Load webpage content
    loader = WebBaseLoader(job_url)
    pg_data = loader.load()

    if not pg_data:
        raise ValueError("No content found on the provided URL.")

    page_content = pg_data.pop().page_content

    # Define the prompt
    prompt_extract = PromptTemplate.from_template(
        """
        ### Scraped Text From Website:
        {page_data}

        ### Instruction:
        Extract detailed job postings from the text above. Provide **only** structured JSON with these fields:
        - `role`: The job title.
        - `experience`: Required experience (if available).
        - `skills`: A list of required skills.
        - `description`: The job description.

        Only include postings that have non-empty values for **role**, **skills**, or **description**.
        Return valid JSON, structured as follows:
        {{
            "job_postings": [
                {{
                    "role": "Software Engineer",
                    "experience": "Entry Level",
                    "skills": ["Skill 1", "Skill 2"],
                    "description": "Job description here."
                }},
                ...
            ]
        }}
        Do not include categories or placeholders without complete details.
        """
    )

    # Execute the LLM chain
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={"page_data": page_content})

    # Clean and validate LLM output
    cleaned_output = clean_json_output(res.content)

    try:
        json_res = json.loads(cleaned_output)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON output from LLM: {cleaned_output}")

    # Filter out incomplete job postings
    valid_postings = [
        job for job in json_res.get("job_postings", [])
        if job.get("role") and (job.get("skills") or job.get("description"))
    ]

    if not valid_postings:
        raise ValueError("No valid job postings found in the provided URL.")

    return {"job_postings": valid_postings}

# Function to format job details for display
def format_job_details(job_details):
    formatted_output = ""
    job_postings = job_details.get("job_postings", [])

    for idx, job in enumerate(job_postings, start=1):
        formatted_output += f"### Job Posting {idx}\n"
        formatted_output += f"- **Role**: {job.get('role', 'N/A')}\n"
        formatted_output += f"- **Experience**: {job.get('experience', 'N/A')}\n"
        formatted_output += f"- **Skills**: {', '.join(job.get('skills', [])) or 'N/A'}\n"
        formatted_output += f"- **Description**: {job.get('description', 'N/A')}\n\n"

    return formatted_output.strip()

# Gradio UI function
def gradio_interface(job_url):
    try:
        job_details = extract_job_details(job_url)
        return format_job_details(job_details)
    except Exception as e:
        return f"An error occurred: {e}"

# Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter Job URL", placeholder="https://example.com/job", lines=1),
    outputs=gr.Markdown(label="Job Details"),
    live=True
)

# Launch Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
