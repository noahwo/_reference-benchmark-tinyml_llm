# %% Set up the environment
import os
import subprocess
import json
import logging
from typing import List
import pandas as pd
import prompt_templates.templates_sketch as templates_sketch
import traceback
import sys
from langsmith import traceable
from tmp.tmp_utils import delete_ino_files
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "default"

global num_run
num_run = 3
traceable_tag = ["sketch_generator", OPENAI_MODEL_NAME, "valid_new_round5"]
traceable_key = {"num_run": num_run}
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/sketch_generator.log")
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# %%
"""Prompt definitions"""

logger.info("Loading prompt templates.")
full_pro_tem = templates_sketch.full_pro_tem

context_pro_tem = templates_sketch.context_pro_tem

error_handling_prompt = PromptTemplate.from_template(
    templates_sketch.error_handling_pro_tem
)

app_spec_prompt = PromptTemplate.from_template(templates_sketch.app_spec_pro_tem)

sketch_guideline_prompt = PromptTemplate.from_template(
    templates_sketch.sketch_guideline_pro_tem
)

task_prompt_fill_specs = PromptTemplate.from_template(
    templates_sketch.task_pro_tem_fill_specs
)
task_pro_tem_sketch = PromptTemplate.from_template(templates_sketch.task_pro_tem_sketch)
chat_history = """ """

# Error handling needs this information, while this info is relatively isolated from func try_and_retry_code_execution()
current_operation = ""

# TODO: Modify the paths defined in files to work universally


def update_chat_history(prompt, response_raw):
    """Synthesize a chat history item from the prompt and response."""
    global chat_history
    chat_history = f'"most_recent_round_of_conversation":[HumanMessage({str(prompt)})]'
    logger.debug(f"Updated chat history.")


# %% Debug the dataset_summary function
def dataset_summary(dataset_path):
    """Return the summary of the dataset as a string to provide inspirations to LLM for data processing suggestion generation."""
    dataframe = pd.read_csv(dataset_path)
    description = str(dataframe.describe().to_json())
    logger.info(f"Dataset summary generated for {dataset_path}")
    return str(
        '"dataset_summary":{'
        + f'"Dataset shape":"{dataframe.shape}","Basic Statistics":"{description}","note":"column names are case sensitive, remember that."'
        + "}"
    )


# print((dataset_summary("data/spam_email/stages/originral_spam.csv")))
# %%


# Create a function to get user input
def get_user_input():
    # logger.info("Getting user input")
    application_name_hinting_its_purpose = "Object Classifier by Color"
    application_description = "Uses RGB color sensor input to Neural Network to classify objects and outputs object class to serial using Unicode emojis."
    board_fullname = "Arduino Nano 33 BLE Sense"
    classification_classes = "Apple, Banana, Orange"
    input_datatype = "np.float32"
    output_datatype = "np.uint8"
    dataset_path = "data/fruit_to_emoji/SampleData/apple.csv"
    logger.info("User input acquired")
    return (
        application_name_hinting_its_purpose,
        application_description,
        board_fullname,
        classification_classes,
        input_datatype,
        output_datatype,
        dataset_path,
    )
    # path to the model header file is fixed


# Create a function to execute the code snippet safely
def execute_code(code):
    """Decoupled local code execution function."""

    original_dir = os.getcwd()

    tmp_dir = os.path.join(original_dir, "compiling")
    tmp_file = os.path.join(tmp_dir, "compiling.ino")
    save_file = os.path.join(tmp_dir, "compiling_valid.ino")
    # Ensure the directory exists
    os.makedirs(tmp_dir, exist_ok=True)
    logger.info(f"Compilation directory:{tmp_dir}")
    with open(tmp_file, "x") as file:
        file.write(code)

    # Change to the tmp_dir directory before executing the command
    original_dir = os.getcwd()
    os.chdir(tmp_dir)
    logger.info("Compiling with the returned sketch...")
    # Try to compile the sketch and handle errors
    try:
        result = subprocess.run(
            # temporaly hardcoded command
            ["arduino-cli", "compile", "--fqbn", "arduino:mbed:nano33ble", "."],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Compilation SUCCEEDED:\n {result.stdout}")
            logger.info(f"Sketch compiled successfully, see sketch {save_file}")
            # with open(save_file, "x") as file:
            #     file.write(code)
            sys.exit(0)
        else:
            logger.error(f"Compilation failed with error: {result.stderr}")
            return result.stderr
    except Exception as e:
        logger.error(f"Exception during code execution.")
        traceback.print_exc()
    finally:
        os.remove(tmp_file)
        os.chdir(original_dir)


@traceable(
    run_type="llm",
    name="retry",
    tags=traceable_tag,
    metadata=traceable_key,
)
# Create a function to handle retries
def retry_code_execution(
    code,
    max_retries=5,
    llm=None,
):
    """General function for local code execution, also responsible for handling re-prompting and re-executing in case of errors."""
    global current_operation
    for i in range(max_retries + 1):
        if i == max_retries:
            logger.error(f"Max retries reached: {max_retries}")
            sys.exit(
                f"An error could not be resolved after {max_retries} retries: \n{error}"
            )
        error = execute_code(code)
        if not error:
            logger.info("Code executed successfully without errors")
            return [None, llm]
        # logger.error(f"Error: {error}")
        logger.info(f"Trying to solve the error... Re-attempt {i + 1}...")
        user_prompt_error_handling = error_handling_prompt.partial(
            executed_code=code,
            error_info=error,
            current_operation=current_operation,
            # chat_history_key_value="{chat_history_key_value}",
        )
        logger.info(f"Re-invoking the LLM...")
        response_raw = llm.invoke(
            user_prompt_error_handling.format(chat_history_key_value=chat_history)
        )
        response = response_raw.content
        try:
            code = response.split("```cpp\n")[1].split("```")[0]
        except IndexError:
            try:
                code = response.split("```ino\n")[1].split("```")[0]
            except IndexError:
                code = None  # or handle the error as needed
        update_chat_history(user_prompt_error_handling, response_raw)
        logger.info(f"Sketch code received again.")
    return [error, llm]


@traceable(
    run_type="llm",
    name="fill_technical_specs",
    tags=traceable_tag,
    metadata=traceable_key,
)
def fill_technical_specs(
    application_name_hinting_its_purpose,
    application_description,
    board_fullname,
    classification_classes_list,
    input_datatype,
    output_datatype,
    dataset_summary_str,
):
    logger.info("Filling technical specifications of the application")
    app_spec_prompt_to_fill = app_spec_prompt.format(
        application_name_hinting_its_purpose=application_name_hinting_its_purpose,
        application_description=application_description,
        board_fullname=board_fullname,
        classification_classes=json.dumps(classification_classes_list),
        input_datatype=input_datatype,
        output_datatype=output_datatype,
        decide_when_generating_code_based_on_given_board_and_application_description="{decide_when_generating_code_based_on_given_board_and_application_description}",
        decide_when_generating_code_based_on_given_data_sample_and_application_description="{decide_when_generating_code_based_on_given_data_sample_and_application_description}",
        guideline="{guideline}",
    )

    task_prompt_fill_specs_formatted = task_prompt_fill_specs.format(
        app_spec_pro_tem=app_spec_prompt_to_fill,
        dataset_summary=dataset_summary_str,
    )

    full_prompt = full_pro_tem.format(
        context_prompt=context_pro_tem,
        user_prompt=task_prompt_fill_specs_formatted,
    )

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

    response_raw = llm.invoke(full_prompt)

    response = response_raw.content
    app_specifications = response.split("```json")[1].split("```")[0]

    update_chat_history(full_prompt, response_raw)
    return app_specifications


@traceable(
    run_type="llm",
    name="main",
    tags=traceable_tag,
    metadata=traceable_key,
)
# Create a main function to orchestrate the code generation process
def main():
    # logging.info("Starting main function.")
    try:
        user_inputs = get_user_input()
        (
            application_name_hinting_its_purpose,
            application_description,
            board_fullname,
            classification_classes,
            input_datatype,
            output_datatype,
            dataset_path,
        ) = user_inputs

        # logging.debug(f"User inputs: {user_inputs}")

        classification_classes_list = [
            c.strip() for c in classification_classes.split(",")
        ]
        dataset_summary_str = dataset_summary(dataset_path)

        app_specifications_filled_raw = fill_technical_specs(
            application_name_hinting_its_purpose,
            application_description,
            board_fullname,
            classification_classes_list,
            input_datatype,
            output_datatype,
            dataset_summary_str,
        )

        global current_operation
        current_operation = "write_ino_sketch_for_the_application"

        app_specifications_with_guideline = app_specifications_filled_raw.replace(
            "{guideline}", str(sketch_guideline_prompt.template)
        )

        task_prompt_sketch_formatted = task_pro_tem_sketch.format(
            app_spec_pro_tem=app_specifications_with_guideline,
            dataset_summary=dataset_summary_str,
        )

        full_prompt_sketch = full_pro_tem.format(
            context_prompt=context_pro_tem,
            user_prompt=task_prompt_sketch_formatted,
        )

        logger.info("Invoking the LLM to generate sketch code.")
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)
        response_raw = llm.invoke(full_prompt_sketch)
        response = response_raw.content

        try:
            sketch_code = response.split("```cpp\n")[1].split("```")[0]
        except IndexError:
            try:
                sketch_code = response.split("```ino\n")[1].split("```")[0]
            except IndexError:
                sketch_code = None

        update_chat_history(full_prompt_sketch, response_raw)
        logger.info(f"Sketch code received.")

        returned_list = retry_code_execution(sketch_code, 6, llm)
        llm = returned_list[1]
        if returned_list[0] is not None:
            error = returned_list[0]
            logger.error(f"Exiting due to failed attempts to solve the error: {error}")
            sys.exit(f"Exiting due to failed 5 tries to solve the error: \n{error}")

        logger.info("Sketch code generation process completed successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
