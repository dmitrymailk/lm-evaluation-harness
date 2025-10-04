import os

tasks = [
    "qa1",
    "qa2",
    "qa3",
    "qa4",
    "qa5",
]

lengths = [
    "0k",
    "1k",
    "2k",
    "4k",
    "8k",
]
BASE_MODEL_TEMPLATE = """
include: ../_babilongv2_common
task: babilongv2_{dataset_split_qa}_{name}_base
metadata:
  dataset_split_qa: {dataset_split_qa}
  name: {name}
  is_instruct: false
"""

BASE_MODEL_UNDER = """
group: {group_name}
task:
{tasks_names}
"""

generated_tasks = []
for task in tasks:
    os.system(f"mkdir -p {task}")
    for length in lengths:
        task_name = f"babilongv2_{task}_{length}_base"
        yaml_name = f"{task}/{task_name}"
        generated_tasks.append(" - " + task_name)
        with open(f"{yaml_name}.yaml", "w") as f:
            new_config = BASE_MODEL_TEMPLATE.format(
                dataset_split_qa=task,
                name=length,
            )
            f.write(new_config)
    for i, length in enumerate(lengths):
        tasks_names = "\n".join(generated_tasks[: i + 1])
        group_name = f"babilongv2_{task}_under_{length}_base"
        with open(f"{task}/{group_name}.yaml", "w") as f:
            new_config = BASE_MODEL_UNDER.format(
                group_name=group_name,
                tasks_names=tasks_names,
            )
            f.write(new_config)
    generated_tasks = []
