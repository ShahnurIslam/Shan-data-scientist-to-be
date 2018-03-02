---
layout: post
comments: true
published: false
title: Beautiful Jekyll & Cookie Consent implementation
tags:
  Jekyll
---

In May 2011 the EU passed a Cookie Law that affects all cookie enabled websites.  A cookie enabled website now needs to inform a visitor to their site that they are using cookies. If this is your first time on my site you would have seen the below.


![CookieConsent](/img/Cookie Consent.jpg)

I thought this post might be useful to others who have a **Beautiful Jekyll site** that uses **cookies** and live in the EU. 

If the above didn't make any sense let me explain.

### What's a cookie?

A quick explanation of a cookies is that it's a small bit of text that is downloaded onto your device when you visit a site. My wife said it's a horribly misleading term that makes you think of baked goods and should be called something else. 

Cookie Image

Websites can use this data for a myriad of things . As an example a site may use the cookie to track how many times you visited the site, how long you've been there & whether you made any purchases.  Before the cookie law came in, this information was gathered without the user being aware. I won't go into all the details about the cookie law but more information can be found [here](https://www.cookielaw.org/faq/#Whatsthecookielawallabout). 


### What's Beautiful Jekyll?*(Doesn't mean pretty Dr Jekyll!)*

Beautiful Jekyll is a tool I use to create my blog site. It was created by Dean Attali and allows users to create & host their site on Github . I've mainly seen it used for Data Science blogs but could be used for anything. 

### Cookie Consent Implementation


First of all we will need some script that will create the cookie consent pop up on the site. If you go to the site [Cookie Consent](https://cookieconsent.insites.com/download/). You can quickly create a snippet of script and adjust the apperance of how it pops up on your site. 

Once you've tweaked the appearance. Copy the code, go to your github page that you site is based on 
Folder

_updates
head.html

Paste the script and the very bottom of the file. Click Commit and you're done.

Easy peasy!

Hope that helps others and thank you for reading!

Thanks,
Shan


----------

