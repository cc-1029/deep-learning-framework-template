import argparse
import datetime
import pathlib

from utils import common_utils, logging_utils


def get_cli_args():
    parser = argparse.ArgumentParser(description='Deep Learning Template')
    parser.add_argument('-c', '--config', default='conf/config_tf_keras.yml', type=str,
                      help='config file path (default: None)')
    args = parser.parse_args()
    return args


def init_trainer(parser, logger, saved_model_path):
    train_dataloader = parser.init_obj('train_dataloader')
    val_dataloader = parser.init_obj('val_dataloader')
    model = parser.init_obj('models')
    optimizer = parser.init_obj('optimizer')
    loss = parser.init_obj('loss')
    num_epochs = parser['trainers']['num_epochs']
    trainer = parser.init_obj('trainers', logger, model, loss, optimizer, train_dataloader,
                       val_dataloader=val_dataloader, num_epochs=num_epochs)
    trainer.saved_model_path = str(saved_model_path)
    return trainer


def init_logger(logger_path, logger_name):
    logger = logging_utils.get_logger(logger_path, logger_name)
    return logger


if __name__ == '__main__':
    args = get_cli_args()

    ycp = common_utils.YamlConfigParser.from_cli_args(args)
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    saved_model_name = ycp['trainers']['saved_model_name']

    saved_path = pathlib.Path('saved/')
    logger_path = saved_path / 'logs' / f'EXP_{saved_model_name}' / now_str
    saved_model_path = saved_path / 'models' / saved_model_name / now_str

    logger = init_logger(logger_path, saved_model_name)
    trainer = init_trainer(ycp, logger, saved_model_path)
    logger.info('start training')
    trainer.train(use_custom=True)
    logger.info('finish training')
