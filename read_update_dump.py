import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import re,os, sys
import glob, traceback
import seaborn as sns

drug_keywords_map= {
'tobacco' : ['nicotine', 'tobacco', 'cigarette', 'cigarrette', 'cigar', 'bidis', 'snuff'],
'alcohol' : ['liquo', 'beer', 'wine'],
'cannabinoids' : ['marijuana', 'blunt', 'dope', 'ganja', 'grass', 'herb', 'joint', 'bud', 'mary jane', 'pot', 'reefer', 'green', 'trees', 'smoke', 'sinsemilla', 'skunk', 'weed','hashish', 'boom', 'gangster', 'hash', 'hash oil', 'hemp'],
'opioids' : ['heroin', 'smack', 'horse', 'brown sugar', 'dope', 
             # 'H',
             'junk', 'skag', 'skunk', 'white horse', 'China white','opium', 'laudanum', 'paregoric', 'big O', 'black stuff', 'block', 'gum', 'hop'],
'stimulants': ['cocaine', 'hydrochloride', 'blow', 'bump',
               # 'C',
                'candy', 'Charlie', 'coke', 'crack', 'flake', 'rock', 'snow', 'toot','amphetamine', 'Biphetamine', 'Dexedrine', 'bennies', 'black beauties', 'crosses', 'hearts', 'LA turnaround', 'speed', 'truck drivers', 'uppers','methamphetamine', 'Desoxyn','meth', 'ice', 'crank', 'chalk', 'crystal', 'fire', 'glass', 'go fast', 'speed'],
'club_drugs':['mdma', 'ecstasy', 'adam', 'clarity', 'eve', "lover's speed", 'peace', 'uppers','flunitrazepam', 'rohypnol', 'forget-me pill', 'mexican valium', 'r2', 'roach', 'roche', 'roffles', 'roofinol', 'rope', 'rophies','ghb', 'Gamma-hydroxybutyrate', 
              # 'G',
              'Georgia home boy', 'grievous bodily harm', 'liquid ecstasy', 'soap', 'scoop', 'goop', 'liquid X'],
'dissociative_drugs':['ketamine', 'Ketalar SV', 'cat Valium',
                      # 'K',
                      'Special K', 'vitamin K','pcp and analogs', 'phencyclidine' 'angel dust', 'boat', 'hog', 'love boat', 'peace pill','salvia divinorum', 'salvia', 'shepherdess’s herb', 'maria pastora', 'magic mint', 'sally-d','dextromethorphan', 'dxm',  'robotripping', 'robo', 'triple'],
'hallucinogens':['lsd','Lysergic acid diethylamide', 'acid', 'blotter', 'cubes', 'microdot', 'yellow sunshine', 'blue heaven', 'mescaline', 'Buttons', 'cactus', 'mesc', 'peyote', 'psilocybin', 'Magic mushrooms', 'purple passion', 'shrooms', 'little smoke'],
'other_compounds':['anabolic_steroids', 'Anadrol', 'Oxandrin', 'Durabolin', 'Depo-Testosterone', 'Equipoise', 'roids', 'juice', 'gym candy', 'pumpers','inhalants', 'Solvents', 'paint thinners', 'gasoline', 'glues', 'gasses', 'butane', 'propane', 'aerosol propellants', 'nitrous oxide',  'nitrites' ,'isoamyl', 'isobutyl', 'cyclohexyl','laughing gas', 'poppers', 'snappers', 'whippets'],
'prescription_medications':['cns_depressants', 'stimulants', 'opioid pain relievers', 'OxyContin','Oxycodone', 'Vicodin', 'Norco', 'Lortab', 'Hydrocodone', 'Acetaminophen', 'Percocet ', 'Oxycodone', 'Acetaminophen','Tramadol','Codeine','Morphine','Methadone','Demerol', 'meperidine','Acetaminophen','Tylenol', 'Excedrin', 'Vanquish','Aspirin', 'Bayer', 'Bufferin', 'Ecotrin', 'Excedrin', 'Vanquish','Diclofenac', 'Voltaren Gel','Ibuprofen', 'Advil', 'Motrin IB','Naproxen', 'Aleve']
}

columns_to_check = list(drug_keywords_map.keys())  

def read_csv(year):
    all_df = []
    months = [1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12]
    #months = [1, 2]
    for month in months:
        mo = f"{month:02d}"
        files = glob.glob('/data2/julina/scripts/tweets/' + year+ '/'+mo+'/pred/dm/*.csv')
        dfs = []
        for csv_file in files:
            try:
                # print(csv_file)
                a = pd.read_csv(csv_file)
                a = a.loc[:, ~a.columns.str.match('Unnamed')]
                a = a.loc[:, ~a.columns.str.match('label')]
                a = a.loc[:, ~a.columns.str.match('text_y')]
                dfs.append(a)
            except:
                print(f' failed for file : {csv_file}')
                pass
        mo_df = pd.concat(dfs, ignore_index=True)
        mo_df['date'] = pd.to_datetime('2020-' + mo)
        print(mo, '--> ', mo_df.shape)
    
        all_df.append(mo_df)
    dff = pd.concat(all_df, ignore_index=True)
    print('-'*30, 'Before data size:', dff.shape)
    return dff

def do(year):
    dff = read_csv(year)
    for keyword, words in drug_keywords_map.items():
        pattern = fr'\b(?:{"|".join(words)})\b'
        dff[keyword] = dff['text'].str.contains(pattern, case=False).astype(int)
    dff['drug_type'] = dff[columns_to_check].apply(lambda row: [col for col, val in zip(columns_to_check, row) if val == 1], axis=1)
    dff.drop(columns_to_check, axis=1, inplace=True)
    print('-'*30, 'Final data size:', dff.shape)
    dff.to_csv('/data2/julina/scripts/tweets/cleaned_data_by_year/'+ year+'.csv')


##usage
## python3 vader_demo.py /data2/julina/scripts/tweets/2020/ 
if __name__ == "__main__":
    try:
        year = sys.argv[1]
        print(':' * 50, )
        do(year)
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  