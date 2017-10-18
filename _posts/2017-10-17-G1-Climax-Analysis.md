---
layout: post
comments: true
published: true
title: G1 Climax
subtitle: Combining Wrestling with Python
tags:
  data science
  Python
---

Here's a post  that is slightly different to the other posts that I've done so far. Rather than having a step by step tutorial on how to do something. The below allows me to grow my python skills and explain a few things about Python


People who work in Data Science tend to work in programming languages R or Python. I've been learning both languages over the last two years and at the moment I'm trying to get better at my Python. One of things people say is to get better at data science is to work on data that you have an interest in. 

One of my main passions in life has been wrestling (which my wife says is lame).  I've watched wrestling for the last 19 years which when you write down is actually quite scary. In this case, I decided do some analysis on Japanese Wrestling! In particular, the G1 Climax tournament.

![G127](/img/G1271.jpg)

The G1 Climax is an annual Japanese wrestling tournament that's held every summer over 20 days. Rather than a simple knockout tournament it's a league made up of two blocks (named A block & B block) each consisting of 20 wrestlers. The winner of each block then face off and the winner goes to headline wrestle kingdom in January.



Let's start off with importing all the relevant libraries which will be updated as we go


```python
import pandas as pd #standard package to work with dataframes
import numpy as np
import itertools # We use this to do some permutations to work out potential matches
import matplotlib.pyplot as plt
import seaborn as sns
import math
import decimal
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
sns.set_style('white')
```

Here are some potential things to look at:
Who on average has highest match rating?
Does Dave Meltzer's rating correlate with match time or any other variables?

Let's start off by reading the csv file with the wrestler's names and which block they're in.


```python
df = pd.read_csv("G1 Competitors.csv") #Using pandas read_csv, attached is the file
```


```python
df.info() #use the dataframe propetry .info() to give us information on the dataframe
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 2 columns):
    A Block    10 non-null object
    B Block    10 non-null object
    dtypes: object(2)
    memory usage: 240.0+ bytes
    

The information above tells us that the data frame only contains 10 entries with no null objects which is good to see. We can just type in **df** into the console and see what the whole data frame looks like. Of course if we had more rows we wouldn't be able to do this and would use *df.head()*


```python
df
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A Block</th>
      <th>B Block</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hiroshi Tanahashi</td>
      <td>Kazuchika Okada</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Togi Makabe</td>
      <td>Toru Yano</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tomohiro Ishii</td>
      <td>Satoshi Kojima</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hirooki Goto</td>
      <td>Michael Elgin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YOSHI-HASHI</td>
      <td>Juice Robinson</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bad Luck Fale</td>
      <td>Tama Tonga</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Yuji Nagata</td>
      <td>SANADA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Zack Sabre Jr.</td>
      <td>EVIL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Kota Ibushi</td>
      <td>Minoru Suzuki</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tetsuya Naito</td>
      <td>Kenny Omega</td>
    </tr>
  </tbody>
</table>
</div>



I noticed that some of the wrestler's name have capitals while others don't. Let's convert all the names to upper case.
We use the data frame method applymap which applies a function to every cell in a data frame and rather than writing a whole function we will just create one inside in the applymap also known as an anonymous function.


```python
df = df.applymap(lambda x: x.upper()) 
df # Let's print the dataframe to see if that's worked
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A Block</th>
      <th>B Block</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIROSHI TANAHASHI</td>
      <td>KAZUCHIKA OKADA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TOGI MAKABE</td>
      <td>TORU YANO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TOMOHIRO ISHII</td>
      <td>SATOSHI KOJIMA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIROOKI GOTO</td>
      <td>MICHAEL ELGIN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YOSHI-HASHI</td>
      <td>JUICE ROBINSON</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BAD LUCK FALE</td>
      <td>TAMA TONGA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>YUJI NAGATA</td>
      <td>SANADA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ZACK SABRE JR.</td>
      <td>EVIL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>KOTA IBUSHI</td>
      <td>MINORU SUZUKI</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TETSUYA NAITO</td>
      <td>KENNY OMEGA</td>
    </tr>
  </tbody>
</table>
</div>



Let's now create the block tables for each block. First let's separate the wrestler from data frames **df** in two data frames called **A_Block** and **B_Block**


```python
A_Block = pd.DataFrame(df['A Block'])
A_Block.columns = ['Wrestler']
B_Block = pd.DataFrame(df['B Block'])
B_Block.columns = ['Wrestler']
```

Ok we've now got two separate data frames called *A_Block* & *B_Block*. Let's add the usual fields you'd see in any sports league **'Matches','Wins','Losses','Draws'**. 

I'm also going to include **'Match Time'** as the total time a wrestler has competed. I also plan to add another column called **'DMR'**, which stands for Dave Meltzer Rating. Dave Meltzer is a 30 year wrestling journalist whose ratings of matches are out of five and is held in high standing in the wrestling community. By default, I've set these to NaN


```python
A_Block['Matches'] = np.NAN
A_Block['Wins'] = np.NAN
A_Block['Losses'] = np.NAN
A_Block['Draws'] = np.NAN
A_Block['Points'] = np.NAN
A_Block['Match_Time'] = np.NAN
A_Block['DMR'] = 0.00
B_Block['Matches'] = np.NAN
B_Block['Wins'] = np.NAN
B_Block['Losses'] = np.NAN
B_Block['Draws'] = np.NAN
B_Block['Points'] = np.NAN
B_Block['Match_Time'] = np.NAN
B_Block['DMR'] = 0.00
```


```python
A_Block.head(1)#lets use head to see if the table has been set up correctly
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler</th>
      <th>Matches</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Draws</th>
      <th>Points</th>
      <th>Match_Time</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIROSHI TANAHASHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
B_Block.head(1)#B block
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler</th>
      <th>Matches</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Draws</th>
      <th>Points</th>
      <th>Match_Time</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KAZUCHIKA OKADA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



How many potential matches per block? This is a form of permutations, well actually combinations as the order doesn't matter in this case. We can use the combinations formula below where my ***n=10*** & ***r=2***


![Combination_equation](/img/Combination_equation.gif)



```python
math.factorial(10)/(math.factorial(2)*math.factorial(10-2))
```




    45.0



Ok we have a total of 45 possible matches for each block.

I was going to write a custom function to list all these possible matches but luckily the python community is so large someone has already done this for me! The function I'm going to use is within the python library itertools. We will use the function combinations as we're looking for all unique combinations. The combination function takes a series as a parameter so we can't simply pass in the data frame and we need to select the Wrestler column as a series.

Using the combination function, I'll create two new data frames called **A_matches** and **B_matches**


```python
A_matches = pd.DataFrame.from_records(list(itertools.combinations(A_Block['Wrestler'],2)), columns = ['Wrestler 1', 'Wrestler 2'])
B_matches = pd.DataFrame.from_records(list(itertools.combinations(B_Block['Wrestler'],2)), columns = ['Wrestler 1', 'Wrestler 2'])
A_matches.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler 1</th>
      <th>Wrestler 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TOGI MAKABE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TOMOHIRO ISHII</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIROSHI TANAHASHI</td>
      <td>HIROOKI GOTO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIROSHI TANAHASHI</td>
      <td>YOSHI-HASHI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIROSHI TANAHASHI</td>
      <td>BAD LUCK FALE</td>
    </tr>
  </tbody>
</table>
</div>



Ok we've got all the potential matches for each block. Let's add a few more fields to these data frames. Let's add the fields **Winner, Loser, Match_Time, Draw, DMR**. We've also added a match counter field labelled **Match**


```python
A_matches['Winner'] = np.NAN
B_matches['Winner'] = np.NAN
A_matches['Loser'] = np.NAN
B_matches['Loser'] = np.NAN
A_matches['Match'] = 1
B_matches['Match'] = 1
B_matches['Match_Time'] = np.NAN
A_matches['Match_Time'] = np.NAN
B_matches['Draw'] = False
A_matches['Draw'] = False
B_matches['DMR'] = 0.00
A_matches['DMR'] = 0.00
```


```python
A_matches.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler 1</th>
      <th>Wrestler 2</th>
      <th>Winner</th>
      <th>Loser</th>
      <th>Match</th>
      <th>Match_Time</th>
      <th>Draw</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TOGI MAKABE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TOMOHIRO ISHII</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIROSHI TANAHASHI</td>
      <td>HIROOKI GOTO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIROSHI TANAHASHI</td>
      <td>YOSHI-HASHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIROSHI TANAHASHI</td>
      <td>BAD LUCK FALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
B_matches.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler 1</th>
      <th>Wrestler 2</th>
      <th>Winner</th>
      <th>Loser</th>
      <th>Match</th>
      <th>Match_Time</th>
      <th>Draw</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KAZUCHIKA OKADA</td>
      <td>TORU YANO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KAZUCHIKA OKADA</td>
      <td>SATOSHI KOJIMA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KAZUCHIKA OKADA</td>
      <td>MICHAEL ELGIN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KAZUCHIKA OKADA</td>
      <td>JUICE ROBINSON</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KAZUCHIKA OKADA</td>
      <td>TAMA TONGA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Ok we now have a table of the potential matches and all the additional fields. I decided to write two custom functions, one that updates the matches data frames with the results and the other to update the block table. I didn't want to type out the full names either as the spellings of Japanese names can be quite hard. So the function does a partial match and prints out the names it's matched as a check


```python
def updateresults(df,winner,loser, time, DMR, draw = False):#We want to find the index for the match for these two wrestlers
    winner = winner.upper() #Tidy up the names
    loser = loser.upper()
    # Search for them in the results table and return their position
    #I really should add an error handler but I'm being lazy and this function is a onetime use
    w1 = list(df[df['Wrestler 1'].str.contains(winner)].index) 
    w2 = list(df[df['Wrestler 2'].str.contains(winner)].index)
    w3 = w1 + w2
    if not w1:
        winner = df.iloc[w2[0],1]
    else:
        winner = df.iloc[w1[0],0]
    
    l1 = list(df[df['Wrestler 1'].str.contains(loser)].index)
    l2 = list(df[df['Wrestler 2'].str.contains(loser)].index)
    l3 = l1 + l2
    if not l1:
        loser = df.iloc[l2[0],1]
    else:
        loser = df.iloc[l1[0],0]
    i = list(set(w3).intersection(l3))
    i = i[0]
    print(winner + ',' +loser)
    df.iloc[i,5] = time
    df.iloc[i,7] = DMR
    #if it's draw add none to winner & loser and turn the draw value to True
    if draw == False:
        df.iloc[i,2] = winner
        df.iloc[i,3] = loser
    else:
        df.iloc[i,2] = 'None'
        df.iloc[i,3] = 'None'
        df.iloc[i,6] = True       
    updatetable()
```


  ```python
def updatetable():# Function to update the table standings after match results
    global A_Block
    global B_Block
    cols = ['Matches','Wins','Losses','Draws','Points']
    
    A_Block['Wins'] = A_Block['Wrestler'].map(A_matches['Winner'].value_counts())
    A_Block['Losses'] = A_Block['Wrestler'].map(A_matches['Loser'].value_counts())
    x = A_matches.loc[A_matches.Draw == True, ['Wrestler 1','Wrestler 2']]
    Draws = (pd.concat([x['Wrestler 1'],x['Wrestler 2']]))
    A_Block['Draws'] = A_Block['Wrestler'].map(Draws.value_counts())
    A_Block = A_Block.fillna(0)#replace the nan's with zeroes so we can tally later
    A_Block['Matches'] = A_Block['Wins'] + A_Block['Losses'] + A_Block['Draws']
    A_Block['Points'] = A_Block['Wins']*2 + A_Block['Draws']
    a=A_matches[['Wrestler 1','DMR']]
    b=A_matches[['Wrestler 2','DMR']]
    a.columns = ['Wrestler','DMR']
    b.columns = ['Wrestler','DMR']
    c = [a,b]
    d = pd.concat(c)
    ratings  = d.groupby('Wrestler').mean().reset_index()
    A_Block['DMR'] = A_Block['Wrestler'].map(ratings.set_index('Wrestler')['DMR'])
    A_Block = A_Block.sort_values(['Points'],ascending=False).reset_index(drop=True)
    A_Block[cols] = A_Block[cols].applymap(np.int64)
    
    
    
    B_Block['Wins'] = B_Block['Wrestler'].map(B_matches['Winner'].value_counts())
    B_Block['Losses'] = B_Block['Wrestler'].map(B_matches['Loser'].value_counts())
    x =B_matches.loc[B_matches.Draw == True, ['Wrestler 1','Wrestler 2']]
    Draws = (pd.concat([x['Wrestler 1'],x['Wrestler 2']]))
    B_Block['Draws'] = B_Block['Wrestler'].map(Draws.value_counts())    
    B_Block = B_Block.fillna(0)#replace the nan's with zeroes so we can tally later
    B_Block['Matches'] = B_Block['Wins'] + B_Block['Losses'] + B_Block['Draws']
    B_Block['Points'] = B_Block['Wins']*2 + B_Block['Draws']
    a=B_matches[['Wrestler 1','DMR']]
    b=B_matches[['Wrestler 2','DMR']]
    a.columns = ['Wrestler','DMR']
    b.columns = ['Wrestler','DMR']
    c = [a,b]
    d = pd.concat(c)
    ratings  = d.groupby('Wrestler').mean().reset_index()
    B_Block['DMR'] = B_Block['Wrestler'].map(ratings.set_index('Wrestler')['DMR'])
    B_Block = B_Block.sort_values(['Points'],ascending=False).reset_index(drop=True)
    B_Block[cols] = B_Block[cols].applymap(np.int64)
```
  


As a test let's update our data frames with results from day 1. I was surprised to see Zack Sabre win over Tanahashi considering he's like the Japan John Cena. If you didn't understand that I don't think you'd be reading this far!


```python
updateresults(A_matches, 'YOSHI', 'Nagata', '16:29',4.25)# Day 1
updateresults(A_matches, 'Fale', 'Togi', '09:25',3.50)
updateresults(A_matches, 'Hirooki', 'Tomohiro', '13:43',4.25)
updateresults(A_matches, 'Zack', 'Tanahashi', '17:18',4.25)
updateresults(A_matches, 'Tetsuya', 'Kota', '24:41',4.75)
```

    YOSHI-HASHI,YUJI NAGATA
    BAD LUCK FALE,TOGI MAKABE
    HIROOKI GOTO,TOMOHIRO ISHII
    ZACK SABRE JR.,HIROSHI TANAHASHI
    TETSUYA NAITO,KOTA IBUSHI
    
Let's check out our **A_matches** dataframe to see if the matches were updated

```python
A_matches
```

<details>
  <summary><b>A_matches <i>CLick to expand</i></b></summary>
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler 1</th>
      <th>Wrestler 2</th>
      <th>Winner</th>
      <th>Loser</th>
      <th>Match</th>
      <th>Match_Time</th>
      <th>Draw</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TOGI MAKABE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TOMOHIRO ISHII</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIROSHI TANAHASHI</td>
      <td>HIROOKI GOTO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIROSHI TANAHASHI</td>
      <td>YOSHI-HASHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIROSHI TANAHASHI</td>
      <td>BAD LUCK FALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HIROSHI TANAHASHI</td>
      <td>YUJI NAGATA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HIROSHI TANAHASHI</td>
      <td>ZACK SABRE JR.</td>
      <td>ZACK SABRE JR.</td>
      <td>HIROSHI TANAHASHI</td>
      <td>1</td>
      <td>17:18</td>
      <td>False</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HIROSHI TANAHASHI</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HIROSHI TANAHASHI</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TOGI MAKABE</td>
      <td>TOMOHIRO ISHII</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TOGI MAKABE</td>
      <td>HIROOKI GOTO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TOGI MAKABE</td>
      <td>YOSHI-HASHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TOGI MAKABE</td>
      <td>BAD LUCK FALE</td>
      <td>BAD LUCK FALE</td>
      <td>TOGI MAKABE</td>
      <td>1</td>
      <td>09:25</td>
      <td>False</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>13</th>
      <td>TOGI MAKABE</td>
      <td>YUJI NAGATA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TOGI MAKABE</td>
      <td>ZACK SABRE JR.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TOGI MAKABE</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>TOGI MAKABE</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TOMOHIRO ISHII</td>
      <td>HIROOKI GOTO</td>
      <td>HIROOKI GOTO</td>
      <td>TOMOHIRO ISHII</td>
      <td>1</td>
      <td>13:43</td>
      <td>False</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TOMOHIRO ISHII</td>
      <td>YOSHI-HASHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TOMOHIRO ISHII</td>
      <td>BAD LUCK FALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TOMOHIRO ISHII</td>
      <td>YUJI NAGATA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>TOMOHIRO ISHII</td>
      <td>ZACK SABRE JR.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>TOMOHIRO ISHII</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>TOMOHIRO ISHII</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>HIROOKI GOTO</td>
      <td>YOSHI-HASHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>HIROOKI GOTO</td>
      <td>BAD LUCK FALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>HIROOKI GOTO</td>
      <td>YUJI NAGATA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HIROOKI GOTO</td>
      <td>ZACK SABRE JR.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>HIROOKI GOTO</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>HIROOKI GOTO</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30</th>
      <td>YOSHI-HASHI</td>
      <td>BAD LUCK FALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>31</th>
      <td>YOSHI-HASHI</td>
      <td>YUJI NAGATA</td>
      <td>YOSHI-HASHI</td>
      <td>YUJI NAGATA</td>
      <td>1</td>
      <td>16:29</td>
      <td>False</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>32</th>
      <td>YOSHI-HASHI</td>
      <td>ZACK SABRE JR.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>YOSHI-HASHI</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>34</th>
      <td>YOSHI-HASHI</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>35</th>
      <td>BAD LUCK FALE</td>
      <td>YUJI NAGATA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>36</th>
      <td>BAD LUCK FALE</td>
      <td>ZACK SABRE JR.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>37</th>
      <td>BAD LUCK FALE</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>38</th>
      <td>BAD LUCK FALE</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>39</th>
      <td>YUJI NAGATA</td>
      <td>ZACK SABRE JR.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>40</th>
      <td>YUJI NAGATA</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>41</th>
      <td>YUJI NAGATA</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>42</th>
      <td>ZACK SABRE JR.</td>
      <td>KOTA IBUSHI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ZACK SABRE JR.</td>
      <td>TETSUYA NAITO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>44</th>
      <td>KOTA IBUSHI</td>
      <td>TETSUYA NAITO</td>
      <td>TETSUYA NAITO</td>
      <td>KOTA IBUSHI</td>
      <td>1</td>
      <td>24:41</td>
      <td>False</td>
      <td>4.75</td>
    </tr>
  </tbody>
</table>
</div>

</details>


<br/>

From the above this seems to have worked and our **A_matches** data frame has now been updated with the results from Day 1. Let's check if the **A_Block** data frame has been updated too.


```python
A_Block
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler</th>
      <th>Matches</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Draws</th>
      <th>Points</th>
      <th>Match_Time</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YOSHI-HASHI</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.472222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BAD LUCK FALE</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.388889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIROOKI GOTO</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.472222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ZACK SABRE JR.</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.472222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TETSUYA NAITO</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.527778</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HIROSHI TANAHASHI</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.472222</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TOGI MAKABE</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.388889</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TOMOHIRO ISHII</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.472222</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YUJI NAGATA</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.472222</td>
    </tr>
    <tr>
      <th>9</th>
      <td>KOTA IBUSHI</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.527778</td>
    </tr>
  </tbody>
</table>
</div>

We can see above the *Win,Loss, Draw, Points & DMR* fields have all been updated. Now let's update all our data frames from all 18 days of competition.



  
  ```python
  updateresults(B_matches,'Juice','Satoshi', '11:48',3.5) # Day 2
  updateresults(B_matches,'Tonga','Elgin', '13:46',3.25)
  updateresults(B_matches,'Sanada','Evil','15:48',4.00)
  updateresults(B_matches,'Okada','Yano', '10:31',3.25)
  updateresults(B_matches,'Omega','Suzuki','21:24',4.75)
  
  updateresults(A_matches, 'Hirooki', 'Nagata', '15:02',4.50)# Day3
  updateresults(A_matches,'ISHII', 'Togi', '15:51',4.50)
  updateresults(A_matches,'Ibushi', 'Zack', '15:51',4.50)
  updateresults(A_matches,'Tanaha', 'Fale', '11:05',3.50)
  updateresults(A_matches,'Tetsuya', 'Yoshi', '22:19',4.25)
  
  updateresults(B_matches, 'Yano', 'Kojima','09:12',1.50) # Day 4
  updateresults(B_matches, 'Evil', 'Juice','11:46',4.00)
  updateresults(B_matches, 'Suzuki', 'Sanada','11:22',3.75)
  updateresults(B_matches, 'Omega', 'Tama','11:42',3.50)
  updateresults(B_matches, 'Okada', 'Elgin','25:49',4.75)
  
  updateresults(A_matches, 'Zack', 'Yoshi','11:48',3.50) # Day 5
  updateresults(A_matches, 'Hiroshi', 'YUji', '14:47',4.50)
  updateresults(A_matches, 'Fale', 'Tetsuya', '11:55',2.50)
  updateresults(A_matches, 'Kota', 'Ishii', '17:14',4.50)
  updateresults(A_matches, 'Togi', 'Hirooki','16:55',4.00)
  
  updateresults(B_matches, 'Elgin', 'Kojima','13:09',3.75) # Day 6
  updateresults(B_matches, 'Evil', 'Tonga','10:27',2.75)
  updateresults(B_matches, 'Suzuki', 'Juice', '11:23',3.25)
  updateresults(B_matches, 'Omega', 'Yano', '11:31',1.00)
  updateresults(B_matches, 'Okada', 'Sanada', '20:49',4.25)
  
  updateresults(A_matches, 'Ishii', 'Yoshi', '15:43',3.75) #Day 7
  updateresults(A_matches, 'Zack', 'Fale', '09:02',2.50)
  updateresults(A_matches, 'Togi', 'Kota', '13:20',4.00)
  updateresults(A_matches, 'Tetsuya', 'Nagata', '15:16',4.00)
  updateresults(A_matches, 'Hiroshi', 'hirooki', '01:22',3.25)
  
  updateresults(B_matches, 'Tonga', 'Juice','10:36',2.75)# Day 8
  updateresults(B_matches, 'Sanada', 'Toru', '04:33',3.00)
  updateresults(B_matches, 'Evil', 'Suzuki', '08:38',3.50)
  updateresults(B_matches, 'Okada', 'Kojima', '15:26',4.25)
  updateresults(B_matches, 'Elgin', 'Omega', '24:39',4.75)
  
  updateresults(A_matches, 'Togi', 'Nagata', '10:45',3.50)# Day 9
  updateresults(A_matches, 'Fale', 'Kota', '11:37',3.50)
  updateresults(A_matches, 'Hirooki', 'Zack','10:10',3.50)
  updateresults(A_matches, 'Hiroshi', 'Yoshi', '13:34',3.75) 
  updateresults(A_matches, 'Ishii', 'Tetsuya', '20:58',4.50)
  
  updateresults(B_matches, 'Evil', 'Yano', '01:33',0.50) # Day 10
  updateresults(B_matches, 'Suzuki', 'Tama','10:22',3.50)
  updateresults(B_matches, 'Sanada', 'Elgin', '15:06',4.00)
  updateresults(B_matches, 'Omega', 'Kojima','12:42',3.50)
  updateresults(B_matches, 'Okada', 'Juice', '20:29',3.50)
  
  updateresults(A_matches, 'Yoshi', 'Fale', '10:21',3.00)# Day 11
  updateresults(A_matches, 'Zack', 'Togi', '09:30',2.75)
  updateresults(A_matches, 'Ishii', 'Nagata', '13:51',4.50)
  updateresults(A_matches, 'Naito', 'Goto', '13:30',4.00)
  updateresults(A_matches, 'Ibushi', 'Hiroshi', '20:40',4.75) 
  
  updateresults(B_matches, 'Sanada', 'Juice', '13:48',3.00) # Day 12
  updateresults(B_matches, 'Yano', 'Elgin', '02:58',1.00)
  updateresults(B_matches, 'Suzuki', 'Kojima', '10:13',3.00)
  updateresults(B_matches, 'Okada', 'Tonga','11:22',3.25)
  updateresults(B_matches, 'Omega', 'Evil','23:33',4.00)
  
  updateresults(A_matches, 'Ibushi', 'Nagata', '15:54',4.25) # Day 13
  updateresults(A_matches, 'Fale', 'ishii', '11:58',3.50)
  updateresults(A_matches, 'Goto', 'Yoshi', '11:26',3.25)
  updateresults(A_matches, 'Naito', 'Zack', '14:20',3.75)
  updateresults(A_matches, 'Tanahashi', 'Togi', '13:34',3.50)
  
  updateresults(B_matches, 'Yano', 'Tonga', '03:15',1.00) # Day 14
  updateresults(B_matches, 'Satoshi', 'Sanada', '12:09',3.50)
  updateresults(B_matches, 'Elgin', 'Suzuki', '11:13',3.00)
  updateresults(B_matches, 'Juice', 'Omega','15:36',3.75)
  updateresults(B_matches, 'Evil', 'Okada','22:47',4.25)
  
  updateresults(A_matches, 'Nagata', 'Zack', '15:08',3.50) # Day 15
  updateresults(A_matches, 'Ibushi', 'Yoshi', '14:28',3.50) 
  updateresults(A_matches, 'Fale', 'Goto', '09:34',2.25)
  updateresults(A_matches, 'Naito', 'Makabe', '11:31',3.25)
  updateresults(A_matches, 'Tanahashi', 'Ishii', '23:30',4.5)
  
  updateresults(B_matches, 'Tonga', 'Koji', '10:43',2.75) # Day 16
  updateresults(B_matches, 'Juice', 'Yano', '04:25',1.50)
  updateresults(B_matches, 'Elgin', 'Evil', '11:07',4.00)
  updateresults(B_matches, 'Omega', 'Sanada','15:03',3.75)
  updateresults(B_matches, 'Okada', 'Suzuki','30:00',4.75,True)
  
  updateresults(A_matches, 'Fale', 'Nagata', '11:56',4.25) # Day 17
  updateresults(A_matches, 'Makabe', 'Yoshi', '11:28',3.50) 
  updateresults(A_matches, 'Zack', 'Ishii', '15:22',4.25)
  updateresults(A_matches, 'Goto', 'Ibushi', '11:03',4.25)
  updateresults(A_matches, 'Naito', 'Tanahashi', '26:41',5.00)
  
  updateresults(B_matches, 'Juice', 'Elgin', '11:48',3.75) # Day 18
  updateresults(B_matches, 'Tonga', 'Sanada', '11:59',3.75)
  updateresults(B_matches, 'Yano', 'Suzuki', '06:56',3.00)
  updateresults(B_matches, 'Evil', 'Satoshi','14:23',4.00)
  updateresults(B_matches, 'Omega', 'Okada','24:40',6.00)
```






Let's look at the results tables and see who came out on top!


```python
A_Block
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler</th>
      <th>Matches</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Draws</th>
      <th>Points</th>
      <th>Match_Time</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TETSUYA NAITO</td>
      <td>9</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>14</td>
      <td>0.0</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIROSHI TANAHASHI</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>12</td>
      <td>0.0</td>
      <td>4.111111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAD LUCK FALE</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>12</td>
      <td>0.0</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KOTA IBUSHI</td>
      <td>9</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>0.0</td>
      <td>4.222222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ZACK SABRE JR.</td>
      <td>9</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>0.0</td>
      <td>3.611111</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HIROOKI GOTO</td>
      <td>9</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>0.0</td>
      <td>3.694444</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TOMOHIRO ISHII</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TOGI MAKABE</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>3.611111</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YOSHI-HASHI</td>
      <td>9</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>0.0</td>
      <td>3.638889</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YUJI NAGATA</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>4.138889</td>
    </tr>
  </tbody>
</table>
</div>




```python
B_Block
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wrestler</th>
      <th>Matches</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Draws</th>
      <th>Points</th>
      <th>Match_Time</th>
      <th>DMR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KENNY OMEGA</td>
      <td>9</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>14</td>
      <td>0.0</td>
      <td>3.888889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KAZUCHIKA OKADA</td>
      <td>9</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>13</td>
      <td>0.0</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EVIL</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>12</td>
      <td>0.0</td>
      <td>3.444444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MINORU SUZUKI</td>
      <td>9</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
      <td>3.611111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SANADA</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MICHAEL ELGIN</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>3.583333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JUICE ROBINSON</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>3.222222</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TAMA TONGA</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>2.944444</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TORU YANO</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.750000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SATOSHI KOJIMA</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>3.305556</td>
    </tr>
  </tbody>
</table>
</div>



Looking at a table is fine, but a visual representation is always better. Let's have a look at the block results in graph form, using the Seaborn package


```python

ax= sns.barplot(y='Wrestler', x='Points', data=A_Block, color= 'Red')
ax.set_xlim(0,18)
sns.despine()
ax.set_ylabel('')
ax.set_xlabel('points')
ax.set_title('A Block')
plt.show()
ax= sns.barplot(y='Wrestler', x='Points', data=B_Block,  color= 'Blue')
ax.set_xlim(0,18)
sns.despine()
ax.set_ylabel('') 
ax.set_xlabel('points')
ax.set_title('B Block')
plt.show()
```


![png](/img/chart_1.png)



![png](/img/output_40_1.png)


The winner of the A Block was Kenny Omega and the B Block was Tetsuya Naito who are two of the biggest wrestlers there so not a surprise *(Yes I know wrestling is fake as my wife keeps saying)*

Ok who was the highest rated star of the tournament


```python
Combined = pd.concat([A_Block, B_Block])
Combined['Block'] = np.NAN
Combined.iloc[0:10,8] ="A Block"
Combined.iloc[10:20,8] ="B Block"
Combined = Combined.sort_values(['DMR'],ascending=False).reset_index(drop=True)
ax = sns.barplot(y='Wrestler', x='DMR',data=Combined, color = 'Gold')
sns.despine()
ax.set_xlabel('Average Dave Meltzer Rating')
plt.show()
```


![png](/img/output_43_0.png)


Ok so the top two wrestlers were Tomohiro Ishii and Kazuchika Okada. I wasn't surprised with Okada who's doing amazing this year but Ishii was a surprise. Yano had the worst overall rating which kind of makes sense as he's more of a comedy act and not a competitive wrestler.

One of the potential questions I had was if a good Meltzer rating related to the match length. Let's plot these two against each other. 


```python
C_matches = pd.concat([A_matches, B_matches])
C_matches['Match_Time'] = C_matches['Match_Time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
ax = sns.regplot(y='DMR', x='Match_Time',data=C_matches)
plt.show()
```


![png](/img/output_46_0.png)


OK so some correlation between the two which makes some sense. Obviously this doesn't account for who's in the match and what actually happened in it. Let's see what our equation would look like


```python

lm = LinearRegression()
X = C_matches['Match_Time'].values[:,np.newaxis]
y = C_matches['DMR'].values
lm.fit(X,y)
m = lm.coef_[0]
b = lm.intercept_
print("Our Linear equation is " + 'y = {0} * x + {1}'.format(m, b))
```

    Our Linear equation is y = 0.002155728983767316 * x + 1.8171627906221903
    

Let's see what the mean squared error is and variance score. These calculations tell us how well our model fits the data.



```python
print("Mean squared error: %.2f" % np.mean((lm.predict(X) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lm.score(X, y))
```

    Mean squared error: 0.36
    Variance score: 0.58
    

Ok so in our model, 58% of the variability in Match ratings can be explained using match time. We also got a really low mean squared error which means the model is a really good fit. 

**But like I said this doesn't account for the context of the matches despite the good fit.**

Hopefully this wasn't too complicated but I wanted to do analysis on some data that I was interested in. I'll most likely add to this post in the future as I learn more modelling tools that account for more of the variables present.

Thanks for reading this post if you did! Any feedback or tips as always would be greatly appreciated.

Shan


