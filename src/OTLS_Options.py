import argparse


class OTLS_Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_path', type=str, default='/Volumes/CSE_BME_AXM788/data/University of Washington_3D_Prostate_Path/', help='path to data')
        self.parser.add_argument('--sample_names', type=str, default='S025_hb_A')
        self.parser.add_argument('--downsample_level', type=str, default='2')
        self.parser.add_argument('--prem_data_path', type=str, default='../prem_results/')
        self.parser.add_argument('--output_path', type=str, default='../output/')
        self.parser.add_argument('--slic_no', type=int, default=100000)
        self.parser.add_argument('--slic_compactness', type=float, default=10.)
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt
