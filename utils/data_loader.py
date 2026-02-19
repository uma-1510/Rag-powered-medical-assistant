import pandas as pd
from typing import List, Dict

def load_data(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

if __name__=="__main__":
    csv_path= "data/medquad.csv"
    passages=load_data(csv_path)

