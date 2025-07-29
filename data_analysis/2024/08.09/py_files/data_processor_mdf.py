# # %% Set up the environment
# import os
# import subprocess
# import json
# import logging
# from typing import List
# import pandas as pd
# import prompt_templates.templates_convert as templates_convert
# import traceback
# import sys

# from langchain_community.llms import OpenAI
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.documents import Document
# from langsmith import traceable
# from tmp.tmp_utils import copy_file, delete_csv_files, print_json_as_table

# # Load environment variables from .env file
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "default"

# # Set up logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# file_handler = logging.FileHandler("logs/data_processor.log")
# console_handler = logging.StreamHandler()

# file_handler.setLevel(logging.INFO)
# console_handler.setLevel(logging.INFO)

# formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# global num_run
# num_run =
# traceable_tag = ["data_processing", "valid_new_round2"]
# traceable_key = {"num_run": num_run}

# """Prompt definitions"""

# # %%
# full_prompt = PromptTemplate.from_template(templates_convert.full_pro_tem)
# system_prompt = PromptTemplate.from_template(templates_convert.context_pro_tem)
# data_suggestion_prompt = PromptTemplate.from_template(
#     templates_convert.data_suggestion_pro_tem
# )
# task_prompt = PromptTemplate.from_template(templates_convert.task_pro_tem_dataeng)
# error_handling_prompt = PromptTemplate.from_template(
#     templates_convert.error_handling_pro_tem
# )
# dataset_summary_prompt = PromptTemplate.from_template(
#     templates_convert.dataset_summary_pro_tem
# )
# chat_history = """ """
# current_operation = ""

# python_interpreter = sys.executable
# # %%


# def update_chat_history(prompt, response_raw):
#     """Synthesize a chat history item from the prompt and response."""
#     global chat_history
#     chat_history = f'"most_recent_round_of_conversation":[HumanMessage({str(prompt)}),AIMessage({str(response_raw.content)})]'
#     logger.debug(f"Updated chat history.")


# # %%
# # %%


# def dataset_summary(dataset_path):
#     """Return the summary of the dataset as a string to provide inspirations to LLM for data processing suggestion generation."""
#     try:
#         dataframe = pd.read_csv(dataset_path)
#         description = str(dataframe.describe().to_json())
#         head = str(dataframe.head().to_json())
#         logger.info(f"Dataset summary generated for {dataset_path}")
#         return str(
#             {
#                 "Dataset shape": str(dataframe.shape),
#                 "Dataset descriptive statistics": description,
#                 "Dataset first 5 rows": head,
#                 "note": "column names are case sensitive, remember that.",
#             }
#         )
#     except Exception as e:
#         logger.error(f"Error generating dataset summary for {dataset_path}: {e}")
#         logger.debug(traceback.format_exc())
#         return {}


# @traceable(
#     run_type="llm",
#     name="suggestion_table",
#     tags=traceable_tag,
#     metadata=traceable_key,
# )
# def generate_suggestion_table(
#     purpose: str,  # The purpose of the model input by user
#     dataset_path: str,  # Path to the dataset provided by the user
#     dataset_intro: str,  # Introduction to the dataset provided by the user
# ):
#     """Returns a suggestion table based on the dataset meta-info."""
#     dataset_summary_str = dataset_summary(dataset_path)
#     llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

#     pipeline_prompt_0 = PipelinePromptTemplate(
#         final_prompt=full_prompt,
#         pipeline_prompts=[
#             ("system_prompt", system_prompt),
#             ("user_prompt", data_suggestion_prompt),
#             (
#                 "dataset_summary",
#                 dataset_summary_prompt,
#             ),
#         ],
#     )

#     prompt = pipeline_prompt_0.format(
#         purpose=purpose,
#         dataset_summary=dataset_summary_str,
#         dataset_intro=dataset_intro,
#     )

#     returnable = llm.invoke(prompt)
#     update_chat_history(prompt, returnable)
#     logger.info("Generated suggestion table using LLM.")
#     return json.loads(returnable.content)


# # Create a function to get user input
# def get_user_input():
#     logger.info("Getting user input")
#     purpose = "detect spam emails"
#     copy_file()
#     dataset_path = "./data/spam_email/raw_spam.csv"
#     dataset_intro = """This dataset contains a collection of emails, categorized into two classes: "Spam" and "Non-Spam" (often referred to as "Ham"). These emails have been carefully curated and labeled to aid in the development of spam email detection models. Whether you are interested in email filtering, natural language processing, or machine learning, this dataset can serve as a valuable resource for training and evaluation."""
#     logger.info("User input acquired")
#     return purpose, dataset_path, dataset_intro


# # Create a function to execute the code snippet safely
# def execute_code(code):
#     """Decoupled local code execution function."""

#     tmp_file = "./tmp/tmp1.py"
#     with open(tmp_file, "w") as file:
#         file.write(code)

#     logger.info("Executing code snippet.")
#     # Try to execute the script and handle errors
#     try:
#         result = subprocess.run(
#             [python_interpreter, tmp_file], capture_output=True, text=True
#         )
#         if result.returncode == 0:
#             logger.info("Code executed successfully.")
#             return None
#         else:
#             logger.error(f"Code execution failed: {result.stderr}")
#             return result.stderr
#     finally:
#         # Ensure the script file is deleted regardless of success or failure
#         os.remove(tmp_file)


# @traceable(
#     run_type="llm",
#     name="retry",
#     tags=traceable_tag,
#     metadata=traceable_key,
# )
# # Create a function to handle retries
# def retry_code_execution(
#     code,
#     max_retries=5,
#     llm=None,
# ):
#     """General function for local code execution, also responsible for handling re-prompting and re-executing in case of errors."""
#     global current_operation
#     for i in range(max_retries + 1):
#         if i == max_retries:
#             logger.error(f"Max retries reached: {max_retries}")
#             delete_csv_files()
#             sys.exit(
#                 f"An error could not be resolved after {max_retries} retries: \n{error}\nMax retries reached: {max_retries}"
#             )
#         error = execute_code(code)
#         if not error:
#             logger.info("Code executed successfully without errors")
#             return [None, llm]
#         logger.error(f"Error: {error}")
#         logger.info(f"Trying to solve the error... Attempt {i + 1}...")
#         user_prompt_error_handling = error_handling_prompt.partial(
#             executed_code=code,
#             error_info=error,
#             current_operation=current_operation,
#             extra_req="",  # TODO: refactor and remove this
#         )
#         response_raw = llm.invoke(
#             user_prompt_error_handling.format(chat_history_key_value=chat_history)
#         )
#         response = response_raw.content
#         # update the most recent chat history
#         update_chat_history(user_prompt_error_handling, response_raw)
#         code = response.split("```python")[1].split("```")[0]

#     return [error, llm]


# """The code below belongs to main() function"""


# # Create a main function to orchestrate the data engineering process
# @traceable(
#     run_type="llm",
#     name="main",
#     tags=traceable_tag,
#     metadata=traceable_key,
# )
# def main():
#     python_interpreter = sys.executable
#     (purpose, dataset_path, dataset_intro) = get_user_input()

#     suggestion_table = generate_suggestion_table(purpose, dataset_path, dataset_intro)

#     processing_already_applied = []
#     llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

#     for operation_n, operation_n_explanation in suggestion_table.items():
#         user_prompt_task = task_prompt.format(
#             operation_n=operation_n,
#             operation_n_explanation=operation_n_explanation,
#             dataset_path=dataset_path,
#             list_processing_already_applied=str(processing_already_applied),
#         )
#         global current_operation
#         current_operation = '"' + operation_n + '":"' + operation_n_explanation + '"'
#         response_raw = llm.invoke(user_prompt_task + chat_history)
#         response = response_raw.content
#         # Update the mode recent chat history
#         update_chat_history(user_prompt_task, response_raw)
#         code = response.split("```python")[1].split("```")[0]

#         try:
#             error, llm = retry_code_execution(code, max_retries=5, llm=llm)
#         except Exception as e:
#             error = f"Error: {str(e)}"

#         processing_already_applied.append(
#             {str(operation_n): {"error": str(error), "code": code}}
#         )
#         logger.info(f"Operation {operation_n} completed.")

#     logger.info("Data processing process completed:")
#     logger.info(print_json_as_table(suggestion_table))


# if __name__ == "__main__":
#     main()
