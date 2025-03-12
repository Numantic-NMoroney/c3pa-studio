# [2503] n8
#

import json, os, sys

sys.path.insert(0, "data")


def configure_app():
    with open("data/c3pa_studio_path.json") as file:
        mount_path = json.load(file)['mount_path']

    has_term = os.environ.get('TERM') == 'xterm-256color'
    if has_term:
        app_path = './c3pa-app/testing'
    else:
        app_path = os.environ.get('MOUNT_PATH', mount_path)

    with open(app_path + "/c3pa_studio_config.json") as file:
        cfg = json.load(file)

    return mount_path, app_path, cfg


def get_count(app_path, name_txt):
    count_path = os.path.join(app_path, name_txt)
    count = ''
    with open(count_path, "r") as file:
        count = file.readline()
    return count

