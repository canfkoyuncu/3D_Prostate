import configparser

from src.VolumeH5 import Volume


def main():
    ''' ----------- load data --------------- '''
    prost_vol = Volume()
    print("volume has been loaded.")
    prost_vol.print_info()


if __name__ == "__main__":
    main()
