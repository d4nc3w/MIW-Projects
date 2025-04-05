import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True, help='Output csv file')
    parser.add_argument('-i', '--input', nargs='+', default=[], help='Input csv files')
    parser.add_argument('-s', '--long_name', nargs=2, default=[])

    return parser.parse_args()

def main(args: argparse.Namespace)-> None:
    print(f'Output file: {args.output}')
    print(f'Input files: {args.input}')
    dataset = pd.DataFrame()
    for input_file in args.input:
        input_data = pd.read_csv(input_file, sep=',')
        name = input_data['Name'][0]
        dataset[name] = input_data['Low']
        dataset.to_csv(args.output, index=False)

if __name__ == '__main__':
    main(parse_args())



