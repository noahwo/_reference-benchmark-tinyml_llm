import os
import sys
from dotenv import load_dotenv
import subprocess
import time

sys.path.append(
    os.path.join(os.path.dirname(__file__), "/home/han/Projects/tinyml-autopilot/src")
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "/home/han/Projects/tinyml-autopilot")
)


from src.factories.llm_factory import LLMFactory
from src.processors.model_converter import ModelConverter
from src.processors.data_processor import DataProcessor
from src.processors.sketch_generator import SketchGenerator
import concurrent.futures


def run_batch_test(num_run, testee, benchmarking=False):
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")

    llm_strategy = LLMFactory.create_llm(
        "openai", api_key=openai_api_key, model_name=openai_model_name
    )

    match testee:
        case "data":
            processor = DataProcessor(llm_strategy, num_run, benchmark=benchmarking)
        case "convert":
            processor = ModelConverter(llm_strategy, num_run, benchmark=benchmarking)
        case "sketch":
            processor = SketchGenerator(llm_strategy, num_run, benchmark=benchmarking)
    processor.run()


def main():
    # data: DataProcessor, convert: ModelConverter, sketch: SketchGenerator
    testee = "data"
    num_runs = 30
    benchmarking = True

    for i in range(num_runs):
        # Use subprocess to run the batch test in a separate process
        subprocess.run(
            [
                sys.executable,  # Use the current Python interpreter
                __file__,  # The current script file
                str(i),  # Pass the current run index
                testee,  # Pass the testee type
                str(benchmarking),  # Pass the benchmarking flag
            ]
        )
        time.sleep(20)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments are passed, run the batch test with those arguments
        run_batch_test(int(sys.argv[1]), sys.argv[2], sys.argv[3] == "True")
    else:
        # Otherwise, run the main function
        main()
