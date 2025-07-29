# %% [markdown]
# ## 1. Setting up
#

import json

# %%
# ref: https://langfuse.com/docs/query-traces
import os
from datetime import datetime

import pandas as pd
from langfuse import Langfuse

LANGFUSE_SERVICE_PUBLIC_KEY = "pk-lf-8b62fd88-54c8-4b3d-a5d9-791ba8a7e90a"
LANGFUSE_SERVICE_SECRET_KEY = "sk-lf-f6a07b0f-1304-492a-aca6-322e264eb313"
LANGFUSE_SERVICE_HOST = "https://cloud.langfuse.com"

LANGFUSE_LOCAL_PUBLIC_KEY = "pk-lf-f24eaab4-afd5-4895-8d52-580a242b99a4"
LANGFUSE_LOCAL_SECRET_KEY = "sk-lf-c6b7cebb-6877-4b71-8d3f-f1be40a046b4"
LANGFUSE_LOCAL_HOST = "http://localhost:3000"

langfuse_secret_key = "sk-lf-c6b7cebb-6877-4b71-8d3f-f1be40a046b4"
langfuse_public_key = "pk-lf-f24eaab4-afd5-4895-8d52-580a242b99a4"
langfuse_host = "http://localhost:3000"

"""Define paths"""
# print(os.getcwd())
parent_dir = os.path.dirname(os.getcwd())

date = "03.18"
date = os.path.basename(parent_dir)
tex_dir = os.path.join(parent_dir, "tex")
processed_data_dir = os.path.join(parent_dir, "processed_data")
raw_export_dir = os.path.join(parent_dir, "raw_export")
ipynb_dir = os.path.join(parent_dir, "ipynb")

"""Define session_id"""
# session_id="qwen2.5-coder_f4d4_dp_batch"
session_id_list = [
    # "qwen2.5-coder:14b_1bb2_mc_batch",
    # "qwen2.5-coder:14b_1bb2_dp_batch",
    # "deepseek-r1:14b_60e0_mc_batch",
    # "phi4_6fca_sg_batch",
    # "llama3.1_da8e_sg_batch"
    # "qwen2.5-coder:14b_b154_sg_batch"
    "llama3.1_02c0_sg_batch"
]

# %% [markdown]
# ## 1.9 Pretty print `fetch_traces_response`
#

# %%
import json
from datetime import datetime
from typing import Any, Dict, List


class TraceWithDetails:
    def __init__(
        self,
        id: str,
        timestamp: datetime,
        name: str,
        input: Any,
        output: Any,
        session_id: str,
        release: Any,
        version: Any,
        user_id: Any,
        metadata: Dict[str, Any],
        tags: List[str],
        public: bool,
        html_path: str,
        latency: float,
        total_cost: float,
        observations: List[str],
        scores: List[Any],
        externalId: Any,
        bookmarked: bool,
        projectId: str,
        environment: str,
        createdAt: str,
        updatedAt: str,
    ):
        self.id = id
        self.timestamp = timestamp
        self.name = name
        self.input = input
        self.output = output
        self.session_id = session_id
        self.release = release
        self.version = version
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.public = public
        self.html_path = html_path
        self.latency = latency
        self.total_cost = total_cost
        self.observations = observations
        self.scores = scores
        self.externalId = externalId
        self.bookmarked = bookmarked
        self.projectId = projectId
        self.environment = environment
        self.createdAt = createdAt
        self.updatedAt = updatedAt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "session_id": self.session_id,
            "release": self.release,
            "version": self.version,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "tags": self.tags,
            "public": self.public,
            "html_path": self.html_path,
            "latency": self.latency,
            "total_cost": self.total_cost,
            "observations": self.observations,
            "scores": self.scores,
            "externalId": self.externalId,
            "bookmarked": self.bookmarked,
            "projectId": self.projectId,
            "environment": self.environment,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
        }

    def pretty_print(self) -> None:
        """
        Converts the trace details to a dictionary and pretty prints the JSON representation.
        """
        trace_dict = self.to_dict()
        pretty_json = json.dumps(trace_dict, indent=4)
        print(pretty_json)


# %% [markdown]
# ## 2.1 Export raw data
#
# Langfuse added a limit of 20 API invocations per minute. https://langfuse.com/faq/all/api-limits
#

import json

# %%
# ALTERNATIVE TO 2.
import os
from datetime import datetime
from time import sleep

from langfuse import Langfuse
from tqdm import tqdm

langfuse = Langfuse(
    secret_key=LANGFUSE_SERVICE_SECRET_KEY,
    public_key=LANGFUSE_SERVICE_PUBLIC_KEY,
    host=LANGFUSE_SERVICE_HOST,
)

API_invok_count = 0
query_range_num_run = {"start": 0, "end": 1}


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            data = obj.__dict__.copy()
            if "observations" in data:
                data["observations"] = [
                    fetch_observation_data(obs) for obs in data["observations"]
                ]

            return data
        return super().default(obj)


def fetch_observation_data(observation_id):
    """
    Fetches observation data from Langfuse and returns its dictionary representation.
    """
    print(f"Fetching observation data for {observation_id}...")
    global API_invok_count
    if API_invok_count >= 19:
        print("Waiting for 60 seconds to fetch observation data...")
        for _ in tqdm(range(60), desc="Progress", unit="s"):
            sleep(1)
        API_invok_count = 0

    observation_response = langfuse.fetch_observation(observation_id)
    API_invok_count += 1

    return observation_response.data.dict()


def fetch_and_save_complete_data(session_id_list, raw_export_dir):
    """
    Fetches complete trace data for each session ID and saves it to JSON files.

    Parameters:
        session_id_list (list): List of session IDs to process.
        raw_export_dir (str): Directory path to save raw JSON files.
    """

    def save_complete_data(session_id):
        global API_invok_count
        if API_invok_count >= 15:
            print("Waiting for 60 seconds to fetch traces...")
            for _ in tqdm(range(60), desc="Progress", unit="s"):
                sleep(1)
            API_invok_count = 0

        fetch_traces_response = langfuse.fetch_traces(session_id=session_id)
        API_invok_count += 1

        print(f"Fetching traces for session {session_id}...")
        # Create directories if they don't exist
        os.makedirs(raw_export_dir, exist_ok=True)

        # Save complete data to JSON file
        # if session_id.startswith("da0a"):
        #     session_id = "phi4_" + session_id
        raw_path = os.path.join(raw_export_dir, f"raw_{session_id}.json")
        with open(raw_path, "w") as f:
            json.dump(fetch_traces_response, f, cls=CustomJSONEncoder, indent=2)

        print(f"Raw JSON saved to: {raw_path}")

    for session_id in session_id_list:
        save_complete_data(session_id)


fetch_and_save_complete_data(session_id_list, raw_export_dir)

# %% [markdown]
# ## 2.2 Trim data
#
# Here also intercept the runs with fatal errors that need to be excluded from the analysis.
#

import json

# %%
import os
from datetime import datetime

skipped_traces = []


def process_existing_observation(observation):
    """
    Processes an existing observation dictionary by trimming unwanted keys.
    """
    unwanted_observation_keys = [
        "completionStartTime",
        "metadata",
        "timeToFirstToken",
        "createdAt",
        "usageDetails",
        "usage",
        "projectId",
        "unit",
        "updatedAt",
        "version",
        "parentObservationId",
        "promptId",
        "promptName",
        "promptVersion",
        "modelId",
        "inputPrice",
        "outputPrice",
        "totalPrice",
        "modelParameters",
        "input",
        "output",
    ]

    # If observation is a dictionary containing observation data
    if isinstance(observation, dict):
        trimmed_observation = {
            k: v for k, v in observation.items() if k not in unwanted_observation_keys
        }
        return trimmed_observation
    return observation


def trim_data(data):
    """
    Recursively trims the data structure.
    """

    if isinstance(data, dict):
        # Process the current dictionary
        unwanted_trace_keys = [
            "release",
            "version",
            "user_id",
            "public",
            "html_path",
            "scores",
            "bookmarked",
            "projectId",
            "externalId",
            "page",
            "limit",
            "total_pages",
        ]

        # If this is a trace that contains observations, check for fatal errors
        if "observations" in data:
            # Check for SPAN observations with fatal errors before processing
            skip_trace = False
            for obs in data["observations"]:
                if isinstance(obs, dict) and obs.get("name").startswith("error"):
                    status_message = obs.get("statusMessage", "")
                    ob_name = obs.get("name")
                    print(f"SPAN {ob_name}: {status_message}")

                    if "Fatal error" in status_message:
                        print(f"Found Fatal error in SPAN observation, skipping trace")
                        skip_trace = True
                        skipped_traces.append(data["name"])
                        break

            if skip_trace:
                return None  # Signal to skip this trace

        # Create a new dictionary with wanted keys and recursively process values
        trimmed_data = {}
        for key, value in data.items():
            if key not in unwanted_trace_keys:
                if key == "observations":
                    # Special handling for observations
                    trimmed_data[key] = [
                        process_existing_observation(obs) for obs in value
                    ]
                elif isinstance(value, (dict, list)):
                    # Recursively process nested structures
                    trimmed_data[key] = trim_data(value)
                else:
                    trimmed_data[key] = value

        return trimmed_data

    elif isinstance(data, list):
        # Recursively process each item in the list
        processed_items = []
        for item in data:
            processed_item = trim_data(item)
            if processed_item is not None:  # Only add items that weren't filtered out
                processed_items.append(processed_item)
        return processed_items

    else:
        # Return non-dict, non-list values as is
        return data


def read_and_trim_data(session_id_list, raw_export_dir, trimmed_export_dir):
    """
    Reads complete data from JSON files, trims the data, and saves the trimmed data to new JSON files.
    """
    os.makedirs(trimmed_export_dir, exist_ok=True)

    for session_id in session_id_list:
        try:
            if session_id.startswith("da0a"):
                session_id = "phi4_" + session_id
            # Read raw data
            raw_path = os.path.join(raw_export_dir, f"raw_{session_id}.json")
            with open(raw_path, "r") as f:
                data = json.load(f)

            # Process and trim the data
            trimmed_data = trim_data(data)

            # If the entire data was filtered out (unlikely but possible)
            if trimmed_data is None:
                print(
                    f"All traces in session {session_id} were filtered due to fatal errors"
                )
                continue

            # Save trimmed data
            trimmed_path = os.path.join(
                trimmed_export_dir, f"trimmed_{session_id}.json"
            )
            with open(trimmed_path, "w") as f:
                json.dump(trimmed_data, f, indent=2)

            print(
                f"Successfully processed and saved trimmed data for session {session_id}"
            )

            # Optional: Verify trimming worked
            # print(f"Verifying trimmed data for session {session_id}...")
            # verify_trimming(trimmed_path)

        except Exception as e:
            print(f"Error processing session {session_id}: {str(e)}")


def verify_trimming(trimmed_path):
    """
    Verifies that the trimmed data doesn't contain unwanted keys.
    """
    with open(trimmed_path, "r") as f:
        trimmed_data = json.load(f)

    unwanted_keys = [
        "release",
        "version",
        "user_id",
        "public",
        "html_path",
        "scores",
        "bookmarked",
        "projectId",
        "externalId",
        "page",
        "limit",
        "total_pages",
        "completionStartTime",
        "metadata",
        "usageDetails",
        "timeToFirstToken",
        "createdAt",
        "completionTokens",
        "promptTokens",
        "projectId",
        "unit",
        "updatedAt",
        "version",
        # "statusMessage",
        "parentObservationId",
        "promptId",
        "promptName",
        "promptVersion",
        "modelId",
        "inputPrice",
        "outputPrice",
        "totalPrice",
        "calculatedInputCost",
        "calculatedOutputCost",
        "calculatedTotalCost",
    ]

    def check_keys(obj):
        if isinstance(obj, dict):
            for key in obj.keys():
                if key in unwanted_keys:
                    print(f"Warning: Found unwanted key '{key}' in trimmed data")
            for value in obj.values():
                check_keys(value)
        elif isinstance(obj, list):
            for item in obj:
                check_keys(item)

    check_keys(trimmed_data)
    print("Verification complete")


# Usage example:
read_and_trim_data(session_id_list, raw_export_dir, raw_export_dir)
print(f"Total {len(skipped_traces)} traces skipped. They are {skipped_traces}")

# %% [markdown]
# ## 3. Generate CSV files from JSON
#

# %%
import traceback

import pandas as pd


def json_to_csv(session_id):
    """
    Convert JSON trace data to CSV format with aggregated metrics.

    Args:
        session_id (str): Identifier for the session to process
    """

    def extract_observation_details(observations, trace_id):
        """Extract and aggregate metrics from observations"""
        metrics = {
            "status": None,
            "latency": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0,
            "input_cost": 0,
            "output_cost": 0,
        }

        # Process GENERATION observations
        for obs in (o for o in observations if o["type"] == "GENERATION"):
            metrics["total_tokens"] += obs["totalTokens"]
            metrics["prompt_tokens"] += obs["promptTokens"]
            metrics["completion_tokens"] += obs["completionTokens"]
            metrics["latency"] += obs["latency"]

            # Add costs if present
            for cost_type in ["Total", "Input", "Output"]:
                key = f"calculated{cost_type}Cost"
                metric_key = cost_type.lower() + "_cost"
                if obs.get(key) is not None:
                    metrics[metric_key] += obs[key]

        # Process SPAN observations for status
        status_indicators = [
            obs["name"]
            for obs in observations
            if obs["type"] == "SPAN" and "start_" not in obs["name"]
        ]

        # Determine status
        success_signals = sum("end_" in name for name in status_indicators)
        failure_signals = sum("failure_signal" in name for name in status_indicators)

        if success_signals + failure_signals > 1:
            raise ValueError(f"Multiple status indicators found in trace {trace_id}")

        metrics["status"] = (
            "success"
            if success_signals
            else "failure" if failure_signals else "unknown"
        )

        metrics["prompt_cost"] = metrics.pop("input_cost")
        metrics["completion_cost"] = metrics.pop("output_cost")
        metrics["latency"] = round(metrics["latency"] / 1000, 2)
        return metrics

    def cal_time(trace):
        time_diff = datetime.fromisoformat(
            trace["updatedAt"].replace("Z", "+00:00")
        ) - datetime.fromisoformat(trace["createdAt"].replace("Z", "+00:00"))
        seconds_diff = time_diff.total_seconds()
        return seconds_diff

    try:

        if session_id.startswith("da0a"):
            session_id = "phi4_" + session_id
        simple_session_id = session_id.rsplit("_", 2)[0]

        # Load JSON data
        trimmed_path = os.path.join(raw_export_dir, f"trimmed_{session_id}.json")
        print(
            f"Processing session {session_id}, simple id {simple_session_id}. Look for {trimmed_path}"
        )
        with open(trimmed_path, "r") as file:
            traces = json.load(file)["data"]

        # Process traces
        rows = [
            {
                "num_run": trace["metadata"]["num_run"],
                "name": trace["name"],
                "trace_id": trace["id"],
                "batch_id": trace["session_id"],
                "latency": cal_time(trace),
                # "latency": round(trace["latency"], 2),
                **extract_observation_details(
                    trace["observations"],
                    trace["id"],
                ),
                "tags": trace["tags"],
            }
            for trace in traces
        ]
        # print(rows)
        # Create and save DataFrame
        df = pd.DataFrame(rows).sort_values("num_run")

        output_dir = os.path.join(processed_data_dir, f"{simple_session_id}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"clean_{session_id}.csv")

        print(output_path)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved CSV to: {output_path}")

    except FileNotFoundError as e:
        print(
            f"FileNotFoundError: For session {session_id} not found. Looked for {trimmed_path}\nError info: \n{e}\n\nTraceback: {traceback.format_exc()}"
        )
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in input file for session {session_id}")
    except Exception as e:
        print(f"Error processing session {session_id}: {str(e)}")


# Example usage
for session_id in session_id_list:
    json_to_csv(session_id)

# %% [markdown]
# ## Code below is archived
#

# %%
# """Simply calculate success rate"""


# def cal_success_rate(session_id):

#     end_signal_count = 0
#     failure_signal_count = 0
#     # Function to print the name of each observation
#     with open(f"{raw_export_dir}/trimmed_{session_id}.json", "r") as file:
#         data = json.load(file)["data"]
#     for i in data:

#         observations = i["observations"]

#         for observation in observations:
#             # print(type(observation))
#             for key, value in observation.items():
#                 # print(f"{key}: {value}")
#                 for key, value in value.items():
#                     # print(f"{key}: {value}")
#                     if key == "name":
#                         if "end_" in value:

#                             end_signal_count += 1
#                         if "failure_signal" in value:

#                             failure_signal_count += 1

#     print(f"Session ID: {session_id}")
#     total_count = end_signal_count + failure_signal_count
#     if total_count > 0:
#         success_rate = end_signal_count / total_count
#         print(f"Success rate: {success_rate:.4f}")
#     else:
#         print("Success rate: N/A (no signals found)")
#     print(f"Passed:\t{end_signal_count}\nFailed:\t{failure_signal_count}")
#     print(
#         ""
#         if total_count == 30
#         else "Number of ending signals does not match the expected number!"
#     )
#     print("-" * 50)


# for session_id in session_id_list:
#     cal_success_rate(session_id)

# %% [markdown]
#

# %%
# def cal_time(start_time, end_time):
#     time_diff = datetime.fromisoformat(
#         end_time.replace("Z", "+00:00")
#     ) - datetime.fromisoformat(start_time.replace("Z", "+00:00"))
#     seconds_diff = time_diff.total_seconds()
#     return seconds_diff


# print(cal_time("2025-01-15T03:31:56.150000+00:00", "2025-01-15T03:32:59.384Z"))

# %%
# """Print the complete structure of exported json file"""
# def print_keys(d, parent_key=''):
#     if isinstance(d, dict):
#         for key, value in d.items():
#             full_key = f"{parent_key}.{key}" if parent_key else key
#             print(full_key)
#             print_keys(value, full_key)
#     elif isinstance(d, list):
#         for i, item in enumerate(d):
#             full_key = f"{parent_key}[{i}]"
#             print_keys(item, full_key)

# # Load JSON data from a file
# with open('fetch_traces_response.json', 'r') as file:
#     data = json.load(file)['data'][0]
# # Print all keys
# print_keys(data)

# %%
