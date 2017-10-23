---
layout: post
comments: true
published: false
title: G1 Climax functions
subtitle: Combining Wrestling with Python
tags:
  data science
  Python
---

The below are the functions I created to update the matches & block results dataframes in my G1 climax post

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


| | Wrestler 1 | Wrestler 2 | Winner | Loser | Match | Match_Time | Draw | DMR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | HIROSHI TANAHASHI | TOGI MAKABE | NaN | NaN | 1 | NaN | False | 0.00 |
| 1 | HIROSHI TANAHASHI | TOMOHIRO ISHII | NaN | NaN | 1 | NaN | False | 0.00 |
| 2 | HIROSHI TANAHASHI | HIROOKI GOTO | NaN | NaN | 1 | NaN | False | 0.00 |
| 3 | HIROSHI TANAHASHI | YOSHI-HASHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 4 | HIROSHI TANAHASHI | BAD LUCK FALE | NaN | NaN | 1 | NaN | False | 0.00 |
| 5 | HIROSHI TANAHASHI | YUJI NAGATA | NaN | NaN | 1 | NaN | False | 0.00 |
| 6 | HIROSHI TANAHASHI | ZACK SABRE JR. | ZACK SABRE JR. | HIROSHI TANAHASHI | 1 | 17:18 | False | 4.25 |
| 7 | HIROSHI TANAHASHI | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 8 | HIROSHI TANAHASHI | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 9 | TOGI MAKABE | TOMOHIRO ISHII | NaN | NaN | 1 | NaN | False | 0.00 |
| 10 | TOGI MAKABE | HIROOKI GOTO | NaN | NaN | 1 | NaN | False | 0.00 |
| 11 | TOGI MAKABE | YOSHI-HASHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 12 | TOGI MAKABE | BAD LUCK FALE | BAD LUCK FALE | TOGI MAKABE | 1 | 09:25 | False | 3.50 |
| 13 | TOGI MAKABE | YUJI NAGATA | NaN | NaN | 1 | NaN | False | 0.00 |
| 14 | TOGI MAKABE | ZACK SABRE JR. | NaN | NaN | 1 | NaN | False | 0.00 |
| 15 | TOGI MAKABE | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 16 | TOGI MAKABE | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 17 | TOMOHIRO ISHII | HIROOKI GOTO | HIROOKI GOTO | TOMOHIRO ISHII | 1 | 13:43 | False | 4.25 |
| 18 | TOMOHIRO ISHII | YOSHI-HASHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 19 | TOMOHIRO ISHII | BAD LUCK FALE | NaN | NaN | 1 | NaN | False | 0.00 |
| 20 | TOMOHIRO ISHII | YUJI NAGATA | NaN | NaN | 1 | NaN | False | 0.00 |
| 21 | TOMOHIRO ISHII | ZACK SABRE JR. | NaN | NaN | 1 | NaN | False | 0.00 |
| 22 | TOMOHIRO ISHII | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 23 | TOMOHIRO ISHII | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 24 | HIROOKI GOTO | YOSHI-HASHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 25 | HIROOKI GOTO | BAD LUCK FALE | NaN | NaN | 1 | NaN | False | 0.00 |
| 26 | HIROOKI GOTO | YUJI NAGATA | NaN | NaN | 1 | NaN | False | 0.00 |
| 27 | HIROOKI GOTO | ZACK SABRE JR. | NaN | NaN | 1 | NaN | False | 0.00 |
| 28 | HIROOKI GOTO | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 29 | HIROOKI GOTO | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 30 | YOSHI-HASHI | BAD LUCK FALE | NaN | NaN | 1 | NaN | False | 0.00 |
| 31 | YOSHI-HASHI | YUJI NAGATA | YOSHI-HASHI | YUJI NAGATA | 1 | 16:29 | False | 4.25 |
| 32 | YOSHI-HASHI | ZACK SABRE JR. | NaN | NaN | 1 | NaN | False | 0.00 |
| 33 | YOSHI-HASHI | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 34 | YOSHI-HASHI | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 35 | BAD LUCK FALE | YUJI NAGATA | NaN | NaN | 1 | NaN | False | 0.00 |
| 36 | BAD LUCK FALE | ZACK SABRE JR. | NaN | NaN | 1 | NaN | False | 0.00 |
| 37 | BAD LUCK FALE | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 38 | BAD LUCK FALE | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 39 | YUJI NAGATA | ZACK SABRE JR. | NaN | NaN | 1 | NaN | False | 0.00 |
| 40 | YUJI NAGATA | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 41 | YUJI NAGATA | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 42 | ZACK SABRE JR. | KOTA IBUSHI | NaN | NaN | 1 | NaN | False | 0.00 |
| 43 | ZACK SABRE JR. | TETSUYA NAITO | NaN | NaN | 1 | NaN | False | 0.00 |
| 44 | KOTA IBUSHI | TETSUYA NAITO | TETSUYA NAITO | KOTA IBUSHI | 1 | 24:41 | False | 4.75 |


From the above this seems to have worked and our **A_matches** data frame has now been updated with the results from Day 1. Let's check if the **A_Block** data frame has been updated too.


```python
A_Block
```

| | Wrestler | Matches | Wins | Losses | Draws | Points | Match_Time | DMR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | YOSHI-HASHI | 1 | 1 | 0 | 0 | 2 | 0.0 | 0.472222 |
| 1 | BAD LUCK FALE | 1 | 1 | 0 | 0 | 2 | 0.0 | 0.388889 |
| 2 | HIROOKI GOTO | 1 | 1 | 0 | 0 | 2 | 0.0 | 0.472222 |
| 3 | ZACK SABRE JR. | 1 | 1 | 0 | 0 | 2 | 0.0 | 0.472222 |
| 4 | TETSUYA NAITO | 1 | 1 | 0 | 0 | 2 | 0.0 | 0.527778 |
| 5 | HIROSHI TANAHASHI | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.472222 |
| 6 | TOGI MAKABE | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.388889 |
| 7 | TOMOHIRO ISHII | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.472222 |
| 8 | YUJI NAGATA | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.472222 |
| 9 | KOTA IBUSHI | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.527778 |

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


