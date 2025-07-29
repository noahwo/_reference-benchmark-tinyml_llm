import subprocess
import os

py_file = (
    "/Users/hann/Projects/MCUrelated/MLOps_by_LLM_from_scratch/model_converter_mdf.py"
)
py_file_2 = "/Users/hann/Projects/MCUrelated/MLOps_by_LLM_from_scratch/mc_batch.py"
for i in range(3, 21):
    os.chdir("/Users/hann/Projects/MCUrelated/MLOps_by_LLM_from_scratch")
    with open(py_file, "r") as file:
        content = file.read()

    new_content = content.replace("num_run = ", f"num_run = {i}")

    with open(py_file_2, "w") as file:
        file.write(new_content)

    subprocess.run(["/Users/hann/anaconda3/envs/datasci2/bin/python", py_file_2])
