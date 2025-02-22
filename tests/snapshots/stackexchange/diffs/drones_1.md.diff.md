Question
========

Title: Crossfire, Unify VTX settings are being overwritten I have a
crossfire nano hooked up properly to a unify pro32 HV. Everything works
as expected and perfectly except when I use the Crossfire Lite OSD to
change the VTX settings.

When I change the VTX settings, such as channel, I can see the signal
switching in the goggles (goes from perfect image to static) and I can
even change the channel in the goggles to the new channel being blasted
from the VTX and I can see the video. But about 5-10 seconds later, the
VTX settings automatically change back to the initial settings. In the
example above, the channel reverts back to the initial channel and I can
change my goggles back to that channel to see it again.

I've tried hooking up to the TBS agent to switch the channels and that
works as well, but again, it reverts back in 5-10 seconds.

Here's a video to show what's going on:
https://photos.app.goo.gl/p6Dt7A11EWjBS3Vz9

Answer
======

> 6 votes The crossfire OSD has a "my VTX" setting on the transmitter
> settings pane: I had "XF TX LITE \> MY VTX \> ACTIVE" set to "Yes" and
> the channels it would flip back to were set there. It seems the "MY
> VTX" section overrides whatever the unify is trying to do. Once I set
> that to "No" the changes I made stuck.
