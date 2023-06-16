import matplotlib.pyplot as plt
from bioread import read_file
import argparse


def read_acq(file_path):
    data = read_file(file_path)
    for chan in data.channels:
        print(chan)
    plt.figure(figsize=(20, 15))
    for i in range(len(data.channels)):
        chan = data.channels[i]
        if chan.name == "Resp1":
            plt.plot(chan.data, label='{}({})'.format(chan.name, chan.units))
        break
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
               mode='expand', borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', '--file_path', type=str,
                        default='C:/Users/Zed/Desktop/Project-BMFG/binghamtom_data/Data.acq',
                        help='Biosignal File Location')
    args = parser.parse_args()
    read_acq(file_path=args.file_path)
