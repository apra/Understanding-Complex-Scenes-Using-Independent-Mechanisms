import os
import uuid
import sys
import torch

from datetime import datetime


def make_dir(directory_path):
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        Logger.log("Directory {} created.".format(directory_path))
    else:
        Logger.log("Directory {} already exists.".format(directory_path))


# lists all numbers of sequential filenames that already exist
def numbers(dirname, prefix):
    number_list = [-1]
    for filename in os.listdir(dirname):
        if filename.startswith(prefix):
            name, _ = os.path.splitext(filename)
            digits = []
            for character in reversed(name):
                if character.isdigit():
                    digits.append(character)
                else:
                    if character == "c":
                        break
            number_list.append(int("".join(reversed(digits))))
    return number_list


class Logger:
    def __init__(self, params, load=False, version=None):
        self.log_dir = params["log_dir"]
        if load:
            self.save_dir = os.path.join(self.log_dir, params["exp_folder"])
            self.figure_dir = os.path.join(self.save_dir, "figures")
            self.checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
            params = self.load_parameters()
            self.experiment_name = params["exp_name"]
        else:
            self.experiment_name = params["exp_name"]
            if version is not None:
                self.version = "-v" + str(version)
            else:
                self.version = ""
            if params["uuid"]:
                self.version = self.version + str(uuid.uuid4())
            self.time_creation = datetime.now()
            self.save_dir = os.path.join(self.log_dir, self.time_creation.strftime("%Y%m%d%H%M") +
                                         "-" + self.experiment_name +
                                         "-seed" + str(params["seed"]) +
                                         self.version)
            self.figure_dir = os.path.join(self.save_dir, "figures")
            self.checkpoints_dir = os.path.join(self.save_dir, "checkpoints")

            make_dir(self.log_dir)
            make_dir(self.save_dir)
            make_dir(self.figure_dir)
            make_dir(self.checkpoints_dir)
            with open(os.path.join(self.save_dir, "experiment_params.log"), "w+") as f:
                for param_name, param_value in params.items():
                    f.write("{}:{}\n".format(param_name, param_value))
            torch.save(params, os.path.join(self.checkpoints_dir, "params.torch"))

    @staticmethod
    def log(text):
        sys.stdout.write("{}\n".format(text))

    @staticmethod
    def error(text):
        sys.stderr.write("{}\n".format(text))

    @staticmethod
    def cluster_log(text):
        Logger.log(text)
        Logger.error(text)

    def get_sequential_figure_name(self, name):
        count = max(numbers(self.figure_dir, name))
        count += 1

        file_name = os.path.join(self.figure_dir, name + "c" + str(count))

        return file_name

    def store_checkpoint(self, model, optimizer, steps, scheduler=None):
        file_path = os.path.join(self.checkpoints_dir, "state.torch")
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'steps': steps
        }
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()

        torch.save(state, file_path)

    def load_checkpoint(self):
        file_path = os.path.join(self.checkpoints_dir, "state.torch")
        if os.path.isfile(file_path):
            return torch.load(file_path)
        else:
            Logger.error("No checkpoint found at: \"{}\"".format(file_path))
            return None

    def load_parameters(self):
        return torch.load(os.path.join(self.checkpoints_dir, "params.torch"))

    def store_data(self, data, name="temp.log"):
        torch.save(data, os.path.join(self.save_dir, name))
