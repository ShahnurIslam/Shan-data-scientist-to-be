---
layout: post
published: false
comments: true
title: SQL part 1 of ?
tags:
  data science
  sql
---

I had to recall what SQL(Structured Query Language) actually stood for because no one actually calls it 'Structured Query Language'.
People just refer to it as 'see-kwel'.

If you have a lot of historic data that you need to access and interrogate regularly then SQL is probably the way to go.
As a programming language it's probably one of the softer languages and can be learnt by most people. Most of the basic queries are easy to understand if the columns of your data are labelled well. You can usualy decipher what they were asking of the data.

Like the below, the query simply returns all the first names from your customers.
```SQL
SELECT first_name FROM customer
```
However it does have its pitfall. 

-Someone needs to maintain the data

-Whatever goes into the data will come out. Or as programmers like to say 'rubbish in = rubbish out.

-For more complicated questions queries can become quite large.

I'm not going to be teaching SQL in this blog as that's not why I started it. There are plenty of places online that have good tutorals already so just Google 'SQL tutorials'and you should find them.

However I will recommend the site [SQL ZOO](http://sqlzoo.net/). They do have tutorials on there but I recommend the site for the assessment examples. Which asks you questions about a database and you need to write queries to answer them. I alway find the best way to learn is by doing. 

When i first started learning SQL a few years ago, i thought to myself 'I'm good at this' despite only doing simple joins and basic queres. 
I then remember doing a SQL test once in the very early days of my career. A simple question came up and I was stumped:

##### "how would you find a customers first order date?"

Ok...i thought let me sort the colums by order date then customer. But ok how do I get the first order date for each customer. I racked my brain and no matter how I sliced the data it just wouldn't work. 

After the test was done, I looked up the answer and it was so simple I was gobsmacked. All I had to do was use the function _MIN()_
Which basically returns the minimum value in a range which also works with date. 

It made me realise how much I need to learn about SQL. 
Ever since then I've practiced more, read more but still have a long way to go.

Here are a few things I plan to learn and will cover in the next post
-Segmentation
-Windows functions
-Stored procedures
-Rank
