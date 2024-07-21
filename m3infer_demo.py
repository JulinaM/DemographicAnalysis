import csv
import json
from collections import OrderedDict
from m3inference.m3inference import M3Inference
import pprint
import re, os, traceback, sys
import pandas as pd
import json

m3 = M3Inference(use_full_model=False) # see docstring for details

def make_decision(data):
    decisions = {}
    for key, attributes in data.items():
        decision = {}
        for attr, probs in attributes.items():
            decision[attr] = max(probs, key=probs.get)
        decisions[key] = decision
    return decisions


def make_inference(input_folder, output_folder):
    
    for filename in os.listdir(input_folder):
        try:
            if filename.endswith('.csv'):
                input_filepath = os.path.join(input_folder, filename)
                output_filename = filename.replace('.csv', '_dm.csv')
                out_filepath = os.path.join(output_folder, output_filename)
                print(f"Processing {input_filepath} to {out_filepath} ...")
                if os.path.exists(out_filepath):
                    print(f"file for {out_filepath} already exists. Skipping...")
                    continue
                df = pd.read_csv(input_filepath)
                # df=df[:100]
                df['lang']= 'en'
                jsonl_str = df.to_json(orient='records', lines=True).strip()
                lines = jsonl_str.split('\n')
                data = [json.loads(line) for line in lines]
                pred = m3.infer(data) 
                decisions = make_decision(pred)
                ddf = pd.DataFrame.from_dict(decisions, orient='index').reset_index().rename(columns={'index': 'id'})
                joined_df = df.merge(ddf, on='id', how='right')
                joined_df = joined_df.loc[:, ~joined_df.columns.str.match('Unnamed')]
                # joined_df = joined_df.drop(columns=['label', 'text_x', 'lang'])
                joined_df.to_csv(out_filepath)

                print(f"Processed {filename} .")
        except:
            traceback.print_exc()
            pass


#usage:: python3 m3infer_demo.py /data2/julina/scripts/tweets/2020/03/user_csv/jsonl/ /data2/julina/scripts/tweets/2020/03/user_csv/demo 
if __name__ == "__main__":
    try:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        print(':' * 50, input_folder, output_folder, ':' * 50)
        make_inference(input_folder, output_folder)
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  


print(f"M3 infererence have been completed for {input_folder}.")