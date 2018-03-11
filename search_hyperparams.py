"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import itertools

from model.utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/permutations',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/kaggle',
                    help="Directory containing the dataset")
parser.add_argument('--objective', default='sentiment',
                    help="Objective of classifier")


def launch_training_job(parent_dir, data_dir, objective, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir} --objective {objective}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir, objective=objective)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over multiple parameters
    ls_learning_rates = [1e-4, 1e-3, 1e-2]
    ls_reg_strengths = [1e-2, 1e-1]
    ls_embedding_sizes = [50,150]
    ls_lstm_num_units = [25]

    # Define permutations of above hyperparameters
    lsHPs = [ls_learning_rates,ls_reg_strengths,ls_embedding_sizes,ls_lstm_num_units]
    lsHPperms = list(itertools.product(*lsHPs))            

    # for learning_rate in learning_rates:
    counter = 0
    for HPperm in lsHPperms:

        counter += 1
        print(' HP permutation {} of {}'.format(counter,len(lsHPperms)))

        # Define current set of hyperparameters
        learning_rate, reg_strength, embedding_size, lstm_num_units = HPperm


        print(' learning rate:',learning_rate)
        print(' reg_strength:',reg_strength)
        print(' embedding_size:',embedding_size)
        print(' lstm_num_units:',lstm_num_units)

        # Modify the relevant parameter in params
        params.learning_rate = learning_rate
        params.reg_stregnth = reg_strength
        params.embedding_size = embedding_size
        params.lstm_num_units = lstm_num_units

        # Launch job (name has to be unique)
        # job_name = "learning_rate_{}".format(learning_rate)
        # job_name = "HP_permutation_{}".format(counter)
        job_name = "LR_{}_RS_{}_ES_{}_LNU_{}".format(learning_rate,reg_strength,embedding_size,lstm_num_units)
        launch_training_job(args.parent_dir, args.data_dir, args.objective, job_name, params)
