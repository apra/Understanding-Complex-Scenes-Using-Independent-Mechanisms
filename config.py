# License: MIT
# Author: Karl Stelzner

from collections import namedtuple
import os
import argparse
import string
import ast


class HyperParameter:
    safechars = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' .-_[]{}'

    def __init__(self, identifier, param_type, metavar=None, default=None, help=None):
        self.identifier = (''.join([c.lower() for c in identifier if c in self.safechars])) \
            .replace(" ", "_")
        self.type = param_type
        self.metavar = metavar if metavar is not None else self.identifier.upper()
        self.default = default
        self.help = help if help is not None else identifier
        self.value = default
        self.action = None
        if self.type == bool:
            if self.default:
                self.action = "store_false"
            else:
                self.action = "store_true"

    def add_argument_parser(self, parser):
        if self.type == bool:
            parser.add_argument('--' + self.identifier,
                                action=self.action,
                                default=self.default,
                                help=self.help)
        else:
            parser.add_argument('--' + self.identifier,
                                type=self.type,
                                metavar=self.metavar,
                                default=self.default,
                                help=self.help)

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = value


class Configuration:
    def __init__(self, name):
        self.name = name
        self.p = {}  # the parameters of the configuration

    def parse(self, parser):
        for param in self.p.values():
            param.add_argument_parser(parser=parser)
        args, namespace = parser.parse_known_args()
        array_params = ["sigmas_x"]
        for name, value in vars(args).items():
            if name == "configuration":
                continue
            parsed_value = value
            if name in array_params:
                if type(value) is str:
                    parsed_value = ast.literal_eval(value)

            if name in self.p.keys():
                self.p[name] = parsed_value
        return self

    def add_hyper_parameter(self, identifier, param_type, metavar=None, default=None, help=None):
        parameter = HyperParameter(identifier=identifier,
                                   param_type=param_type,
                                   metavar=metavar,
                                   default=default,
                                   help=help)
        self.p[parameter.identifier] = parameter

    def to_string(self):
        s = ""
        for identifier, param in self.p.items():
            s += identifier
            s += ": "
            s += str(param)
            s += "\n"
        return s


class Configurator:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--configuration',
                                 type=str,
                                 metavar='CONFIGURATION',
                                 default="default",
                                 help="Default configuration to be used in this experiment (default, ).")
        self.configurations = {}
        self.add_configuration("common")
        self.add_param_configuration(name_config="common",
                                     identifier="vis_every",
                                     param_type=int,
                                     default=500,
                                     help="Visualize progress every X iterations")
        self.add_param_configuration(name_config="common",
                                     identifier="batch_size",
                                     param_type=int,
                                     default=1,
                                     help="Batch size")
        self.add_param_configuration(name_config="common",
                                     identifier="num_steps",
                                     param_type=int,
                                     default=30000,
                                     help="Number of steps")
        self.add_param_configuration(name_config="common",
                                     identifier="log_dir",
                                     param_type=str,
                                     default="experiments",
                                     help="Destination for all logging of this experiment")
        self.add_param_configuration(name_config="common",
                                     identifier="model_name",
                                     param_type=str,
                                     default="default",
                                     help="Name of the model to use.")
        self.add_param_configuration(name_config="common",
                                     identifier="learning_rate",
                                     param_type=float,
                                     default=5e-4,
                                     help="Learning rate")
        self.add_param_configuration(name_config="common",
                                     identifier="exp_name",
                                     param_type=str,
                                     default="brrrrr",
                                     help="Name of the experiment")
        self.add_param_configuration(name_config="common",
                                     identifier="data_dir",
                                     param_type=str,
                                     default="data",
                                     help="Directory for the training data")
        self.add_param_configuration(name_config="common",
                                     identifier="device",
                                     param_type=str,
                                     default="auto",
                                     help="Device where training takes place")
        self.add_param_configuration(name_config="common",
                                     identifier="seed",
                                     param_type=int,
                                     default=0,
                                     help="Seed for this run")
        self.add_param_configuration(name_config="common",
                                     identifier="uuid",
                                     param_type=bool,
                                     default=True,
                                     help="Disable UUID in the folder name")
        self.add_param_configuration(name_config="common",
                                     identifier="rnd_bkg",
                                     param_type=bool,
                                     default=False,
                                     help="Enable random background")
        self.add_param_configuration(name_config="common",
                                     identifier="dontstore",
                                     param_type=bool,
                                     default=False,
                                     help="Don't store the model checkpoint")
        self.add_param_configuration(name_config="common",
                                     identifier="max_num_objs",
                                     param_type=int,
                                     default=3,
                                     help="Maximum number of objects")
        self.add_param_configuration(name_config="common",
                                     identifier="min_num_objs",
                                     param_type=int,
                                     default=1,
                                     help="Minimum number of objects")
        self.add_param_configuration(name_config="common",
                                     identifier="load",
                                     param_type=bool,
                                     default=False,
                                     help="Whether to load a previous experiment and keep training for num_steps")
        self.add_param_configuration(name_config="common",
                                     identifier="exp_folder",
                                     param_type=str,
                                     default=None,
                                     help="Folder of the experiment where to start from to keep training")
        self.add_param_configuration(name_config="common",
                                     identifier="weight_decay",
                                     param_type=float,
                                     default=0.,
                                     help="Weight decay for the optimizer")

        self.add_configuration("default")
        self.add_param_configuration(name_config="default",
                                     identifier="parallel",
                                     param_type=bool,
                                     default=False,
                                     help="Train using nn.DataParallel")
        self.add_param_configuration(name_config="default",
                                     identifier="num_slots",
                                     param_type=int,
                                     default=4,
                                     help="Number of slots k")
        self.add_param_configuration(name_config="default",
                                     identifier="channel_base",
                                     param_type=int,
                                     default=64,
                                     help="Number of channels used for the first U-Net conv layer")
        self.add_param_configuration(name_config="default",
                                     identifier="input_channels",
                                     param_type=int,
                                     default=3,
                                     help="Channels in the input image")
        self.add_param_configuration(name_config="default",
                                     identifier="num_blocks",
                                     param_type=int,
                                     default=5,
                                     help="Number of blocks in attention U-Net")
        self.add_param_configuration(name_config="default",
                                     identifier="bg_sigma",
                                     param_type=float,
                                     default=0.09,
                                     help="Sigma of the decoder distributions for the first slot")
        self.add_param_configuration(name_config="default",
                                     identifier="fg_sigma",
                                     param_type=float,
                                     default=0.11,
                                     help="Sigma of the decoder distributions for all other slots")
        self.add_param_configuration(name_config="default",
                                     identifier="num_samples",
                                     param_type=int,
                                     default=40000,
                                     help="Number of samples in the dataset")
        self.add_param_configuration(name_config="default",
                                     identifier="pixel_bound",
                                     param_type=bool,
                                     default=True,
                                     help="Disable pixel bound.")
        self.add_param_configuration(name_config="default",
                                     identifier="k_steps",
                                     param_type=int,
                                     default=4,
                                     help="Number of elements in the frame.")

    def add_configuration(self, name):
        self.configurations[name] = Configuration(name)

    def add_param_configuration(self, name_config, identifier, param_type, metavar=None,
                                default=None, help=None):
        self.configurations[name_config].add_hyper_parameter(identifier,
                                                             param_type,
                                                             metavar,
                                                             default,
                                                             help)

    def parse(self):
        args, namespace = self.parser.parse_known_args()
        name_config = args.configuration

        common_args = self.configurations["common"].parse(self.parser)

        if name_config in self.configurations.keys():
            specific_args = self.configurations[name_config].parse(self.parser)
        else:
            specific_args = self.configurations["default"].parse(self.parser)

        return name_config, [
            common_args,
            specific_args
        ]


def load_config():
    config_engine = Configurator()
    config_engine.add_configuration("ECON_sprite")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="beta_loss",
                                          param_type=float,
                                          default=1.,
                                          help="Beta term in the loss, in front of the loss_z_t.")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="gamma_loss",
                                          param_type=float,
                                          default=1.,
                                          help="Gamma term in the loss, in front of the loss_r_t.")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="lambda_competitive",
                                          param_type=float,
                                          default=1.,
                                          help="Lambda term in the competition objective.")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="num_samples",
                                          param_type=int,
                                          default=40000,
                                          help="Number of samples in the dataset")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="bg_sigma",
                                          param_type=float,
                                          default=0.09,
                                          help="Sigma of the decoder distributions for the first slot")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="sigmas_x",
                                          param_type=str,
                                          default="[0.09, 0.11]",
                                          help="Sigma of the decoder distributions for all other slots")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="num_blocks",
                                          param_type=int,
                                          default=5,
                                          help="Number of blocks in attention U-Net")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="num_slots",
                                          param_type=int,
                                          default=4,
                                          help="Number of slots k")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="num_objects",
                                          param_type=int,
                                          default=2,
                                          help="Number of objects in the scene")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="num_experts",
                                          param_type=int,
                                          default=3,
                                          help="Number of experts")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="channel_base",
                                          param_type=int,
                                          default=64,
                                          help="Number of channels used for the first U-Net conv layer")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="input_channels",
                                          param_type=int,
                                          default=3,
                                          help="Channels in the input image")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="scheduler",
                                          param_type=str,
                                          default=None,
                                          help="Type of scheduler: plateau, cosann")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="data_dep_init",
                                          param_type=bool,
                                          default=False,
                                          help="Enable data-dependent initialization")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="punish_factor",
                                          param_type=float,
                                          default=0.1,
                                          help="How much the losing experts are punished")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="competition_temperature",
                                          param_type=float,
                                          default=1.,
                                          help="Temperature in the softmax of the competition "
                                               "between experts. A bigger temperature means that "
                                               "the experts are more equally likely to be picked")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="annealing_start",
                                          param_type=int,
                                          default=0,
                                          help="Starting epoch of the annealing of the parameters "
                                               "beta and gamma in the loss")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="annealing_duration",
                                          param_type=int,
                                          default=200,
                                          help="Number of steps that the annealing phase lasts for")
    config_engine.add_param_configuration(name_config="ECON_sprite",
                                          identifier="latent_dim",
                                          param_type=int,
                                          default=2,
                                          help="Dimension of the latent space in the VAE")

    config_engine.add_configuration("ECON_coinrun")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="num_blocks",
                                          param_type=int,
                                          default=5,
                                          help="Number of blocks in attention U-Net")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="beta_loss",
                                          param_type=float,
                                          default=1.,
                                          help="Beta term in the loss, in front of the loss_z_t.")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="gamma_loss",
                                          param_type=float,
                                          default=1.,
                                          help="Gamma term in the loss, in front of the loss_r_t.")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="lambda_competitive",
                                          param_type=float,
                                          default=1.,
                                          help="Lambda term in the competition objective.")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="bg_sigma",
                                          param_type=float,
                                          default=0.09,
                                          help="Sigma of the decoder distributions for the first slot")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="sigmas_x",
                                          param_type=str,
                                          default="[0.09, 0.11]",
                                          help="Sigma of the decoder distributions for all other slots")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="num_objects",
                                          param_type=int,
                                          default=2,
                                          help="Number of objects in the scene")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="num_experts",
                                          param_type=int,
                                          default=3,
                                          help="Number of experts")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="scheduler",
                                          param_type=str,
                                          default=None,
                                          help="Type of scheduler: plateau, cosann")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="channel_base",
                                          param_type=int,
                                          default=64,
                                          help="Number of channels used for the first U-Net conv layer")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="input_channels",
                                          param_type=int,
                                          default=3,
                                          help="Channels in the input image")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="data_dep_init",
                                          param_type=bool,
                                          default=False,
                                          help="Enable data-dependent initialization")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="punish_factor",
                                          param_type=float,
                                          default=0.1,
                                          help="How much the losing experts are punished")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="competition_temperature",
                                          param_type=float,
                                          default=1.,
                                          help="Temperature in the softmax of the competition "
                                               "between experts. A bigger temperature means that "
                                               "the experts are more equally likely to be picked")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="annealing_start",
                                          param_type=int,
                                          default=0,
                                          help="Starting epoch of the annealing of the parameters "
                                               "beta and gamma in the loss")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="annealing_duration",
                                          param_type=int,
                                          default=200,
                                          help="Number of steps that the annealing phase lasts for")
    config_engine.add_param_configuration(name_config="ECON_coinrun",
                                          identifier="dataset_name",
                                          param_type=str,
                                          default="coirun_dataset_train.hdf5",
                                          help="Name of the file where the dataset is located")

    name_config, cfgs = config_engine.parse()

    params = {"name_config": name_config}
    for cfg in cfgs:
        print(cfg.to_string())
        params.update(cfg.p)

    return params


experiment_parameters = {
    # Training config
    'vis_every': 50,  # Visualize progress every X iterations
    'batch_size': 1,
    'num_epochs': 20,
    'load_parameters': True,  # Load parameters from checkpoint
    'data_dir': "data",  # Directory for the training data
    'parallel': False,  # Train using nn.DataParallel
    # Model config
    'num_slots': 4,  # Number of slots k,
    'num_blocks': 5,  # Number of blocks in attention U-Net
    'channel_base': 64,  # Number of channels used for the first U-Net conv layer
    'bg_sigma': 0.09,  # Sigma of the decoder distributions for the first slot
    'fg_sigma': 0.11,  # Sigma of the decoder distributions for all other slots
    'log_dir': "experiments"
}
sprite_experiment_parameters = experiment_parameters.copy()
clevr_experiment_parameters = experiment_parameters.copy()
clevr_experiment_parameters["num_epochs"] = 200
clevr_experiment_parameters["num_slots"] = 11
clevr_experiment_parameters["num_blocks"] = 6

#
# MonetConfig = namedtuple('MonetConfig', config_options)
#
# sprite_config = MonetConfig(vis_every=50,
#                             batch_size=16,
#                             num_epochs=20,
#                             load_parameters=True,
#                             checkpoint_file='./checkpoints/sprites.ckpt',
#                             save_dir='experiments',
#                             data_dir='data/',
#                             parallel=False,
#                             num_slots=4,
#                             num_blocks=5,
#                             channel_base=64,
#                             bg_sigma=0.09,
#                             fg_sigma=0.11,
#                            )
#
# clevr_config = MonetConfig(vis_every=50,
#                            batch_size=64,
#                            num_epochs=200,
#                            load_parameters=True,
#                            checkpoint_file='/work/checkpoints/clevr64.ckpt',
#                            data_dir=os.path.expanduser('~/data/CLEVR_v1.0/images/train/'),
#                            parallel=True,
#                            num_slots=11,
#                            num_blocks=6,
#                            channel_base=64,
#                            bg_sigma=0.09,
#                            fg_sigma=0.11,
#                            save_dir="experiments"
#                           )
#
#
#
