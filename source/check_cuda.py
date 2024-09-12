import torch
import sys


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
        print("GPU: {}, {}".format(torch.cuda.get_device_name(current_device),
                                   torch.cuda.get_device_properties(current_device)))
        print("Python version:".format(sys.version))
        print("Pytorch version: {}\t cuda version:{}\t number of cuda device: {}".format(torch.__version__,
                                                                                         torch.version.cuda,
                                                                                         torch.cuda.device_count()))
    else:
        print("Device: CPU")
        print("Python version:".format(sys.version))
        print("Pytorch version: {}\t cuda version:{}\t number of cuda device: {}".format(torch.__version__,
                                                                                         torch.version.cuda,
                                                                                         torch.cuda.device_count()))
    return device


if __name__ == "__main__":
    x = torch.rand(5, 3)
    print(x)

    get_device()
