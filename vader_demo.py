from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import pprint
import re, os, traceback, sys
import pandas as pd

analyzer = SentimentIntensityAnalyzer()
def find_sentiment(text):
    vs = analyzer.polarity_scores(text)
    sc = vs['compound']
    emo = 'pos' if sc >= 0.05 else 'neu' if -0.05 < sc < 0.05 else 'neg'
    return sc, emo

def do_singlefile(filename):
    try:
        df = pd.read_csv(filename)
        df = df.loc[:, ~df.columns.str.match('  ')]
        df = df.loc[:, ~df.columns.str.match('label')]
        df = df.loc[:, ~df.columns.str.match('text_y')]
        # df = df[:100]
        print(f"Processing {filename} :: {df.shape}")
        df[['sent_score', 'sentiment']] = df['text'].apply(lambda x: pd.Series(find_sentiment(x)))
        df.to_csv(filename)
        # print(df)
        print(f"Done {filename} :: {df.shape}")
        # break    
    except:
        traceback.print_exc()
        pass
            
def do(input_folder):
    # year_folder = glob.glob('/data2/julina/scripts/tweets/'+year)
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # months = [1]
    # for month in range (1, 13):
    for month in months:
        mo = f"{month:02d}"
        month_folder = input_folder +'/'+ mo +'/pred/dm/'
        print(f"month:: {mo}")
        for filename in os.listdir(month_folder):
            try:
                if filename.endswith('.csv'):
                    input_filepath = os.path.join(month_folder, filename)
                    # output_filename = filename.replace('.csv', '_dm.csv')
                    # out_filepath = os.path.join(output_folder, output_filename)
                    # print(f"Processing {input_filepath}")
                    df = pd.read_csv(input_filepath)
                    df = df.loc[:, ~df.columns.str.match('Unnamed')]
                    df = df.loc[:, ~df.columns.str.match('label')]
                    df = df.loc[:, ~df.columns.str.match('text_y')]
                    # df = df[:100]
                    print(f"Processing {input_filepath} :: {df.shape}")
                    df[['sent_score', 'sentiment']] = df['text'].apply(lambda x: pd.Series(find_sentiment(x)))
                    df.to_csv(input_filepath)
                    # print(df)
                    print(f"Done {filename} :: {df.shape}")
                    # break
            except:
                traceback.print_exc()
                pass


##usage
## python3 vader_demo.py /data2/julina/scripts/tweets/2020/ 
## python3 vader_demo.py /data2/julina/scripts/tweets/cleaned_data_by_year/2021.csv 
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