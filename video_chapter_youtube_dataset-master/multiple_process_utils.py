import math


def subprocess_print_err(err):
    print("Subprocess launch failed!: error:{}".format(err))


def split_data(process_num, data):
    each_process_render_num = math.ceil(len(data) / process_num)
    chunked_data = [data[x : x + each_process_render_num] for x in range(0, len(data), each_process_render_num)]

    return chunked_data

