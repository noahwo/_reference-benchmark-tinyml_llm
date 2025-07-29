# %%
import os
import sys
from dotenv import load_dotenv
import subprocess
import time
import uuid

sys.path.append(
    os.path.join(os.path.dirname(__file__), "/home/han/Projects/tinyml-autopilot/src")
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "/home/han/Projects/tinyml-autopilot")
)

import random
from src.factories.llm_factory import LLMFactory
from src.processors.model_converter import ModelConverter
from src.processors.data_processor import DataProcessor
from src.processors.sketch_generator import SketchGenerator
import concurrent.futures


# %%
def run_batch_test(num_run, testee, batch_id, benchmarking=False, trace_id=None):
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")

    llm_strategy = LLMFactory.create_llm(
        "openai", api_key=openai_api_key, model_name=openai_model_name
    )
    if batch_id == "None":
        batch_id = None

    # Pass trace_id to the processor
    match testee:
        case "data":
            processor = DataProcessor(
                llm_strategy,
                num_run,
                batch_id,
                benchmark=benchmarking,
                trace_id=trace_id,
            )
        case "convert":
            processor = ModelConverter(
                llm_strategy,
                num_run,
                batch_id,
                benchmark=benchmarking,
                trace_id=trace_id,
            )
        case "sketch":
            processor = SketchGenerator(
                llm_strategy,
                num_run,
                batch_id,
                benchmark=benchmarking,
                trace_id=trace_id,
            )
    processor.run()


def main():
    # data: DataProcessor, convert: ModelConverter, sketch: SketchGenerator
    testee = "convert"
    num_runs = 30
    benchmarking = num_runs >= 30

    # Generate a random 4-digit string from the current timestamp
    stamp = "".join(random.sample(str(int(time.time())), 4))
    # batch_id is used to identify the batch_run
    batch_id = f"{stamp}_batch"

    for i in range(num_runs):
        # Generate a unique trace ID for the entire task in this run
        trace_id = str(uuid.uuid4()).split("-")[0]
        print(
            f"#{'='*40}#\n# Running batch test {i+1} of {num_runs} for {testee} #\n#{'='*40}#"
        )
        # Use subprocess to run the batch test in a separate process
        subprocess.run(
            [
                sys.executable,  # Use the current Python interpreter
                __file__,  # The current script file
                str(i),  # Pass the current run index
                testee,  # Pass the testee type
                batch_id,
                str(benchmarking),  # Pass the benchmarking flag
                trace_id,  # Pass the unique trace ID
            ]
        )
        time.sleep(20)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments are passed, run the batch test with those arguments
        run_batch_test(
            int(sys.argv[1]),
            sys.argv[2],
            sys.argv[3],
            benchmarking=sys.argv[4] == "True",
            trace_id=sys.argv[5] if len(sys.argv) > 5 else None,
        )
    else:
        # Otherwise, run the main function
        main()
