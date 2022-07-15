import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data for best run, candidate list and stats')
    parser.add_argument('--dataset', default='100', help='dataset path with all problem sets')
    parser.add_argument('-bestrun', action='store_const', const=True, help='flag to generate bestrun files')

    args = parser.parse_args()
    print(args.bestrun)