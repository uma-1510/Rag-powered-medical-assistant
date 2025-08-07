import pandas as pd
from typing import List,Dict
import json

def load_data(csv_path:str) -> List[Dict]:
    df= pd.read_csv(csv_path)
    passages= df.to_dict(orient="records")
    return passages

def load_json_data(json_path:str):
    with open(json_path,"r") as f:
        data=json.load(f)
    records=[]
    medqa_data= data.get("medqa",{})
    for key,entry in medqa_data.items():
        question=entry.get("question", "")
        options=entry.get("options","")
        answer=entry.get("answer","")

        answer_text= options.get(answer,"") if options else ""

        records.append({"question": question, "answer": answer_text})
        return records



if __name__=="__main__":
    csv_path= "data/medquad.csv"
    json_path= "data/bench_mark.json"
    passages=load_data(csv_path)
    qa_pairs= load_json_data(json_path)
    # print(qa_pairs)
    # print(passages[0])

