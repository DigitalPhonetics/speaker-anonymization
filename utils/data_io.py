def read_kaldi_format(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            splitted_line = line.split()
            if len(splitted_line) == 2:
                data[splitted_line[0].strip()] = splitted_line[1].strip()
            elif len(splitted_line) > 2:
                data[splitted_line[0].strip()] = [x.strip() for x in splitted_line[1:]]
    return data


def save_kaldi_format(data_dict, filename):
    with open(filename, 'w') as f:
        for key, value in sorted(data_dict.items(), key=lambda x: x[0]):
            if isinstance(value, list):
                value = ' '.join(value)
            f.write(f'{key} {value}\n')

