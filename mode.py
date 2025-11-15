#Develop a regression model to predict the total seed dry weight of plants given species, treatment type, panicle count, and ground cover measurements.”

import pandas as pd

# Read the Excel file
file_path = r'C:\Users\Gunjan.Sharma\OneDrive - ONECOM GROUP LIMITED\Desktop\Gunjan\AI ML\project 1\ELGL 2010 SH 042417.xls',sheet=1
data = pd.read_excel(file_path)

# Display the first 3 rows
print(data.head(3))
