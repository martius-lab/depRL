import argparse
import os
import time

import yaml

import wandb
from deprl.vendor.tonic import utils


class WandbProcessor:
    def __init__(self, path):
        self._path = path
        self._current_line_number = 0
        self._last_line_number = 0
        self._last_update = os.path.getmtime(path)
        self._setup_wandb()

    def get_line_number(self, data):
        return len(data["train/episode_score/mean"])

    def _setup_wandb(self):
        data = utils.load_csv_to_dict(self._path)
        self._current_line_number = self.get_line_number(data)
        self._log(data)
        self._last_line_number = self.get_line_number(data)

    def _log(self, data):
        for idx in range(self._last_line_number, self._current_line_number):
            logged_data = {k: v[idx] for k, v in data.items()}
            change = 1
            while change:
                for k, v in logged_data.items():
                    if "/mean" in k:
                        logged_data.pop(k)
                        logged_data[k[:-5]] = v
                        change = 1
                        break
                    change = 0
            wandb.log(logged_data, step=idx)

    def update(self):
        self._last_update, updated = utils.check_if_csv_has_updated(
            self._path, self._last_update
        )
        if updated:
            data = utils.load_csv_to_dict(self._path)
            self._current_line_number = self.get_line_number(data)
            self._log(data)
            self._last_line_number = self.get_line_number(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="None")
    parser.add_argument("--project", type=str, default="None")
    args = parser.parse_args()
    config = yaml.load(
        open(os.path.join(args.path[:-7], "config.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    wandb.init(project=args.project, entity="rlpractitioner", config=config)
    processor = WandbProcessor(args.path)
    while True:
        processor.update()
        time.sleep(100)


if __name__ == "__main__":
    main()
