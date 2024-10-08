import yaml
import os
import re


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


def get_absolute_path(relative_path):
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, relative_path)


def get_exploration_config():
    global_config = get_global_config()
    
    exploration_config = config['exploration']
    exploration_config['input_path'] = get_absolute_path(exploration_config['input_path'])
    exploration_config['output_path'] = get_absolute_path(exploration_config['output_path'])
    
    exploration_config = {**exploration_config, **global_config}
    
    return exploration_config


def get_preprocessing_config():
    global_config = get_global_config()
    
    preprocessing_config = config['preprocessing']
    preprocessing_config['input_path'] = get_absolute_path(preprocessing_config['input_path'])
    preprocessing_config['output_file'] = get_absolute_path(preprocessing_config['output_file'])

    preprocessing_config = {**preprocessing_config, **global_config}    

    return preprocessing_config


def get_feature_extraction_config():
    global_config = get_global_config()
    
    feature_extraction_config = config['feature_extraction']
    feature_extraction_config['input_file'] = get_absolute_path(feature_extraction_config['input_file'])
    feature_extraction_config['output_file'] = get_absolute_path(feature_extraction_config['output_file'])

    feature_extraction_config = {**feature_extraction_config, **global_config}

    return feature_extraction_config


def get_ml_config():
    global_config = get_global_config()
    
    ml_config = config['ml']
    ml_config['input_file'] = get_absolute_path(ml_config['input_file'])

    ml_config = {**ml_config, **global_config}

    return ml_config


def get_dl_config():
    global_config = get_global_config()
    
    dl_config = config['dl']
    dl_config['input_file'] = get_absolute_path(dl_config['input_file'])
    
    dl_config = {**dl_config, **global_config}
    
    return dl_config


def split_channels_to_hemispheres(channels: list):
    left_hemisphere = []
    right_hemisphere = []
    
    for channel in channels:
        channel_number = re.search(r'\d+', channel)
        if channel_number is None:
            continue
        
        if int(channel_number.group()) % 2 == 0:
            right_hemisphere.append(channel)
        else:
            left_hemisphere.append(channel)
    
    return left_hemisphere, right_hemisphere


def get_global_config():
    conf = config['global']
    channels = conf['channels']

    left_hemisphere, right_hemisphere = split_channels_to_hemispheres(channels)
    
    conf['left_hemisphere'] = left_hemisphere
    conf['right_hemisphere'] = right_hemisphere

    return conf
