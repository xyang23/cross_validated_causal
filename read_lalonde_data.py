""" 
Read LaLonde data and store it to a CSV file.

Usage: Download the .txt files of NSW Data Files (Dehejia-Wahha Sample) and PSID and CPS Data Files from [http://users.nber.org/~rdehejia/nswdata2.html] and put them into a \data folder.

"""

import pandas as pd
import glob
import os

# Initialize empty dataframe
lalonde = pd.DataFrame()

# Loop through all .txt files in data/ directory
for file in glob.glob("data/*.txt"):
    # Read file
    df = pd.read_csv(file, sep=r"\s+", header=None)
    df.columns = ['treatment', 'age', 'education', 'black', 'hispanic',
                  'married', 'nodegree', 're74', 're75', 're78']
    
    # Extract group name from filename
    name = os.path.splitext(os.path.basename(file))[0].split("_")
    df["group"] = name[0] if name[1] == "controls" else name[1]
    
    # Append to lalonde
    lalonde = pd.concat([lalonde, df], ignore_index=True)

# Create u74 and u75
lalonde["u74"] = (lalonde["re74"] == 0).astype(int)
lalonde["u75"] = (lalonde["re75"] == 0).astype(int)

for col in ["treatment", "black", "hispanic", "married", "nodegree", "u74", "u75"]:
    lalonde[col] = lalonde[col].astype("category")
    
lalonde.to_csv("lalonde.csv", index=False)
print("Data saved in lalonde.csv")