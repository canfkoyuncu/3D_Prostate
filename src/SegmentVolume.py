import configparser

from src.Volume import Volume


def main():
    ''' ----------- load data --------------- '''
    prost_vol = Volume()
    print("volume has been loaded.")
    prost_vol.print_info()
    #prost_vol.visualize()


if __name__ == "__main__":
    main()
