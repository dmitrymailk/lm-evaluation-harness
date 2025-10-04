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
MODEL_CARD_TEMPLATE = """
include: ../_babilongv2_common
task: babilongv2_{dataset_split_qa}_{name}_{model_type}
metadata:
  dataset_split_qa: {dataset_split_qa}
  name: {name}
  is_instruct: {is_instruct}
"""

MODEL_UNDER_TEMPLATE = """
group: {group_name}
task:
{tasks_names}
"""


def generate_tasks(model_type="base"):
    generated_tasks = []
    is_instruct = str(model_type == "instruct").lower()
    for task in tasks:
        os.system(f"mkdir -p {task}")
        for length in lengths:
            task_name = f"babilongv2_{task}_{length}_{model_type}"
            yaml_name = f"{task}/{task_name}"
            generated_tasks.append(" - " + task_name)
            with open(f"{yaml_name}.yaml", "w") as f:
                new_config = MODEL_CARD_TEMPLATE.format(
                    model_type=model_type,
                    dataset_split_qa=task,
                    name=length,
                    is_instruct=is_instruct,
                )
                f.write(new_config)

        for i, length in enumerate(lengths):
            tasks_names = "\n".join(generated_tasks[: i + 1])
            group_name = f"babilongv2_{task}_under_{length}_{model_type}"
            with open(f"{task}/{group_name}.yaml", "w") as f:
                new_config = MODEL_UNDER_TEMPLATE.format(
                    group_name=group_name,
                    tasks_names=tasks_names,
                )
                f.write(new_config)
        generated_tasks = []


if __name__ == "__main__":
    generate_tasks(model_type="base")
    generate_tasks(model_type="instruct")
