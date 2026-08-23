# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="552" height="375" alt="image" src="https://github.com/user-attachments/assets/60bb747a-43c1-4756-bb15-438049d1df24" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="366" height="221" alt="image" src="https://github.com/user-attachments/assets/8d7422c7-43a3-4e36-b140-42b1b586f635" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="539" height="372" alt="image" src="https://github.com/user-attachments/assets/44d050fe-3438-4d2f-8c96-9f01a57a9cc3" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="585" height="365" alt="image" src="https://github.com/user-attachments/assets/6c95b59f-36ac-4f4f-b803-06b9e11536d8" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="543" height="368" alt="image" src="https://github.com/user-attachments/assets/0fd2914a-5b55-4904-bbc4-48277c119201" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="858" height="360" alt="image" src="https://github.com/user-attachments/assets/f8e9b478-31e7-4d68-bdc6-231599036087" />

```
pip install --upgrade category_encoders
```
<img width="1238" height="473" alt="image" src="https://github.com/user-attachments/assets/ea21f0b6-6dd9-4789-ae14-c2d79fe4641e" />

```
import pandas as pd
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="710" height="360" alt="image" src="https://github.com/user-attachments/assets/e527def7-6669-4a7d-9a65-dd983026c3f0" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
df
```
<img width="841" height="359" alt="image" src="https://github.com/user-attachments/assets/20036d0b-9180-4ee2-aa7f-a77895157b0b" />
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="657" height="370" alt="image" src="https://github.com/user-attachments/assets/2e21880f-c476-4fb2-bd99-43f8c4c4264e" />

```
#FEATURE TRANSFORMATION
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="917" height="439" alt="image" src="https://github.com/user-attachments/assets/4888a992-49a0-41d2-9047-07e35ec9723f" />

```
df.skew()
```
<img width="435" height="116" alt="image" src="https://github.com/user-attachments/assets/38087c1b-b8e8-4652-82c4-9d0f9655a5d1" />

```
#1. LOG TRANSFORMATION
np.log(df["Highly Positive Skew"])
```
<img width="704" height="275" alt="image" src="https://github.com/user-attachments/assets/b8485093-da05-4814-8708-7bf1ed7231ba" />

```
#2. RECIPROCAL TRANSFORMATION
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="695" height="265" alt="image" src="https://github.com/user-attachments/assets/e5ab0b4c-c0bb-4daf-8cc1-01bda8350111" />

```
#4. SQUARE ROOT TRANSFORMATION
np.sqrt(df["Highly Positive Skew"])
```
<img width="715" height="278" alt="image" src="https://github.com/user-attachments/assets/282ee728-b569-4c50-af79-a2bfd649c201" />

```
# 5. SQUARE TRANSFORMATION
np.square(df["Highly Positive Skew"])
```
<img width="680" height="259" alt="image" src="https://github.com/user-attachments/assets/8a92fec1-5d59-4b14-aa8d-1e174249f030" />

```
# POWER TRANSFORMATIONS
# BOX COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1170" height="434" alt="image" src="https://github.com/user-attachments/assets/0ca62705-bdae-4321-a872-edb64da9d76f" />

```
df.skew()
```
<img width="560" height="133" alt="image" src="https://github.com/user-attachments/assets/91fbe277-c3fd-47ec-a170-dec0b7b5d7d9" />
```
# YEO_JOHNSON
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="543" height="171" alt="image" src="https://github.com/user-attachments/assets/fe482d51-c9c2-4091-967d-4c83fff36bc8" />

```
# QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1325" height="447" alt="image" src="https://github.com/user-attachments/assets/f5833b8c-b74b-4275-95f7-7c19fc3cbb03" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="823" height="549" alt="image" src="https://github.com/user-attachments/assets/68fe26d6-61d5-4b1e-a386-174203deda16" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="714" height="545" alt="image" src="https://github.com/user-attachments/assets/ccbd9170-3548-4bab-92f6-3118bb2dae53" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="810" height="541" alt="image" src="https://github.com/user-attachments/assets/54546660-b797-4539-9ac1-514a7362e579" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="783" height="545" alt="image" src="https://github.com/user-attachments/assets/868e1f6e-3038-4aad-a59e-e1d8ef2a7cc0" />



# RESULT:
Therefore all the codes have been run and successfully verified and feature encoding and transformation has been done



