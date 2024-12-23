import os
import json
from torch.utils.data import Dataset

class APPSDataset(Dataset):
    def __init__(self, questions_dir, solutions_dir, max_samples=None):
        """
        Args:
            questions_dir (str): Path to the folder with question .txt files.
            solutions_dir (str): Path to the folder with solution .json files.
            max_samples (int, optional): Limit the dataset size for debugging.
        """
        # Collect and sort all .txt files in questions_dir
        self.question_files = sorted(
            f for f in os.listdir(questions_dir) if f.endswith(".txt")
        )
        # Collect and sort all .json files in solutions_dir
        self.solution_files = sorted(
            f for f in os.listdir(solutions_dir) if f.endswith(".json")
        )

        # The dataset length is the minimum between the two
        self.length = min(len(self.question_files), len(self.solution_files))

        # Store directories
        self.questions_dir = questions_dir
        self.solutions_dir = solutions_dir

        # Optionally limit the number of samples (useful for debugging)
        if max_samples is not None:
            self.length = min(self.length, max_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            A tuple (question_text, solution_text).
            If the .json file has multiple solutions, we pick the first one.
        """
        # Match questionNNNN.txt <--> solutionNNNN.json by index
        question_file = os.path.join(self.questions_dir, self.question_files[idx])
        solution_file = os.path.join(self.solutions_dir, self.solution_files[idx])

        # Read question text
        with open(question_file, "r", encoding="utf-8") as fq:
            question_text = fq.read().strip()

        # Read solutions (often an array of strings in APPS)
        with open(solution_file, "r", encoding="utf-8") as fs:
            solutions = json.load(fs)

        # Pick first solution if multiple exist
        if isinstance(solutions, list) and len(solutions) > 0:
            solution_text = solutions[0].strip()
        else:
            # If no solutions or solutions isn't a list, just convert it to str
            solution_text = str(solutions).strip()

        return question_text, solution_text