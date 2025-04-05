import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True, help='Output csv file')
    parser.add_argument('-i', '--input', nargs='+', default=[], help='Input csv files')

    args = parser.parse_args()
    if not args.input:
        raise ValueError('Cannot start without specified input csv files')
    return args

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

#Todo: add exceptions

# python generate_dataset.py -o dataset.csv -i coin_Monero.csv coin_Dogecoin.csv coin_Bitcoin.csv