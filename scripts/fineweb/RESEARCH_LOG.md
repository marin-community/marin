# Research log and learnings from fineweb

## Errors in readability

1. Sometimes readability-lxml or markdownification is not able to convert an html and throws ValueError
2. Errors inside readability-lxml are mostly Document being empty errors
3. Errors inside markdownification are due to string literal errors
4. Log file to inspect the errors can be found in the logs folder (Search for ValueError)

## Do not use http downloading for common crawl

1. Common Crawl provides two download interfaces. http and s3. Http allows for anonymous downloading, s3 does not.
2. But when you make multiple connection to http server, it tends to fail and only sends partial data.
3. When using requests, it was very hard to catch the error and there was no obvious exception raised.
4. When using smart_open (from dolma), this exception was
   caught `Compressed file ended before the end-of-stream marker was reached`
5. Using both requests and smart_open, the processing completed without any error, however only partial data gets
   processed.
6. When using s3, all links were downloaded without any issue.
7. Dolma, fineweb and other projects should use s3 for downloading common crawl data.

## Use of Ray

1. ChatGPT knows very little about Ray
2. Ray has concept of job, task and actor.
3. Job is what runs on the head node and is submitted using `ray job submit`, this is responsible to scheduling task and
   actors.
4. A job should do as minimum work as possible and the number of jobs should be kept to a minimum.
5. Current setup of fineweb has 1 job which spawns about 50-100k tasks. It runs without any issue.
6. Actors are persistent objects that are stores by Ray and can be accessed by multiple jobs and tasks.

## Fsspec and smart_open

1. Both seems to be very similar and can be used interchangeably.
2. fsspec does seem to be more generic and better for our use case.
3. https://github.com/piskvorky/smart_open/issues/579

## reabability-lxml v/s ReadabiliPy

1. reabability-lxml is a python implementation of readability.js
2. ReadabiliPy offer a way where we can just call `node readability.js` and get the output
3. Since ReadabiliPy calls node for each url it it much slower as compared to readability-lxml
4. However we can start a node server and make calls to the server instead of invoking node each time, which will make
   it faster.
5. Quality:
   1. I looked at initial few example of ReadabiliPy and readability-lxml and the output was very similar.
   2. David looked at readability-lxml recently and seemed to not like the output
6. [TODO] Probably try to start a node server and see how much faster it is.
