---
layout: post
comments: true
published: true
title: Beautiful Jekyll & Cookies implementation[Mini -Post]
tags:
  Jekyll
---

In May 2011 the EU passed a law that affects all cookie enabled websites. A cookie enabled website now needs to inform those who visit thier site that they are using cookies.  

If this is your first time on my site you would have seen the below. 

![CookieConsent](/img/Cookie%20Consent.JPG)

I use cookies to look at traffic on my site and to see how users found my site. My blog is created through Beautiful Jekyll and it took me a little while to implement this. So I made this post in the hopes that it may help others and save them time. 

If the above didn't make any sense let me explain.

## What's a cookie?

A quick explanation of a cookies is that it's a small bit of text that is downloaded onto your device when you visit a site. My wife said it's a horribly misleading term that makes you think of baked goods and should be called something else. 

![Cookies](https://www.askideas.com/media/41/I-Got-99-Cookies-But-A-Bitch-Ate-One-Funny-Cookie-Meme-Picture.jpg){: .center-image }

Websites can use this data for a myriad of things . As an example a site may use the cookie to track how many times you visited the site, how long you've been there & whether you made any purchases.  Before the cookie law came in, this information was gathered without the user being aware. I won't go into all the details about the cookie law but more information can be found [here](https://www.cookielaw.org/faq/#Whatsthecookielawallabout). 


## What's Beautiful Jekyll?*(Doesn't mean pretty Dr Jekyll!)*

Beautiful Jekyll is a website template I used to create my blog site. It was created by Dean Attali and allows users to create & host their site on Github. I've mainly seen it used for Data Science blogs but the contenet could be anything. 

## Cookie Consent Implementation


### **Step 1 Create Cookie Script**
First of all we will need some script that will create a cookie consent pop up. I used [Cookie Consent](https://cookieconsent.insites.com/download/) to create my script. They allow you to customise the theme of the pop up and provide default consent terms.
Feel free to create on your own scirpt if you know how.

### **Step 2 Copy & Paste Script into Beautfiul Jekyll repo**
Once you've created your script, copy the code. We then paste the script into this file **'head.html'** which can be found in the folder ** _updates**  Place it at the very bottom between the {$ end if $} and the </head> tags. See mine below as an example

![example](/img/Head_HTML.PNG)

Click Commit once you're done and it's as easy as that!

Hope that helps others and thank you for reading!

Thanks,
Shan


----------

