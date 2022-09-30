from os.path import join
from datasets import load_dataset
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(
    repo_root(), "datasets_turntaking/dataset/vacation_interview/vacation_interview.py"
)


def load_vacation_interview():
    def process_and_add_name(examples):
        examples["dataset_name"] = "vacation_interview"
        return examples

    dset = load_dataset(DATASET_SCRIPT, name="default", split="train")
    dset = dset.map(process_and_add_name)
    return dset
