import pandas as pd
from ethnicolr import census_ln, pred_census_ln
import re, os, traceback, sys
# https://github.com/appeler/ethnicolr


def find_race(df):
    df_pred = pred_census_ln(df, 'name')
    df_pred.drop(columns=['api', 'black', 'hispanic', 'white'], axis=1, inplace=True)

    return df_pred

def do_singlefile(filename):
    try:
        df = pd.read_csv(filename)
        df = df.loc[:, ~df.columns.str.match('  ')]
        df = df.loc[:, ~df.columns.str.match('label')]
        df = df.loc[:, ~df.columns.str.match('text_y')]
        # df = df.sample(100)
        print(f"Processing {filename} :: {df.shape}")
        df = find_race(df)
        df = df.loc[:, ~df.columns.str.match('Unnamed')]
        op_filename = filename.replace(".csv", "_race.csv")
        df.to_csv(op_filename)
        # print(df)
        print(f"Done {filename} :: {df.shape}")
        # break    
    except:
        traceback.print_exc()
        pass


##usage
## python3 find_race.py /data2/julina/scripts/tweets/2020/ 
if __name__ == "__main__":
    try:
        input_folder = sys.argv[1]
        print(':' * 50, input_folder)
        # do(input_folder)
        do_singlefile(input_folder)
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  