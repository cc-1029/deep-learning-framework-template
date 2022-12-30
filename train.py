import argparse

from base_tf import tf_utils

def get_cli_args():
    args = argparse.ArgumentParser(description='Deep Learning Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    return args


def main():
    print(tf_utils.num_gpus())
    pass


if __name__ == '__main__':
    main()
