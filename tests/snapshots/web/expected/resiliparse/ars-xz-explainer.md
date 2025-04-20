What we know about the xz Utils backdoor that almost infected the world | Ars Technica

Skip to main content
  • Biz & IT
  • Tech
  • Science
  • Policy
  • Cars
  • Gaming & Culture
  • Store
  • Forums
Close

Navigate

  • Store
  • Subscribe
  • Videos
  • Features
  • Reviews
  • RSS Feeds
  • Mobile Site
  • About Ars
  • Staff Directory
  • Contact Us
  • Advertise with Ars
  • Reprints

Filter by topic

  • Biz & IT
  • Tech
  • Science
  • Policy
  • Cars
  • Gaming & Culture
  • Store
  • Forums

Settings

Front page layout


Grid

List

Classic

Site theme

light
dark

Comment activity

    dlwh

    Member since 2018-10-03T02:50:51+00:00

    Posts: 0

    Sign out
    Account settings

    NIGHTMARE SUPPLY CHAIN ATTACK SCENARIO —

    What we know about the xz Utils backdoor that almost infected the world

    Malicious updates made to a ubiquitous tool were a few weeks away from going mainstream.

    Dan Goodin - Apr 1, 2024 6:55 am UTC

    What we know about the xz Utils backdoor that almost infected the world
    Enlarge
    Getty Images

    reader comments

    195

    On Friday, a lone Microsoft developer rocked the world when he revealed a backdoor had been intentionally planted in xz Utils, an open source data compression utility available on almost all installations of Linux and other Unix-like operating systems. The person or people behind this project likely spent years on it. They were likely very close to seeing the backdoor update merged into Debian and Red Hat, the two biggest distributions of Linux, when an eagle-eyed software developer spotted something fishy.

    Further Reading

    Backdoor found in widely used Linux utility targets encrypted SSH connections
    "This might be the best executed supply chain attack we've seen described in the open, and it's a nightmare scenario: malicious, competent, authorized upstream in a widely used library," software and cryptography engineer Filippo Valsorda said of the effort, which came frightfully close to succeeding.

    Researchers have spent the weekend gathering clues. Here's what we know so far.

    What is xz Utils?

    xz Utils is nearly ubiquitous in Linux. It provides lossless data compression on virtually all Unix-like operating systems, including Linux. xz Utils provides critical functions for compressing and decompressing data during all kinds of operations. xz Utils also supports the legacy .lzma format, making this component even more crucial.

    What happened?

    Andres Freund, a developer and engineer working on Microsoft’s PostgreSQL offerings, was recently troubleshooting performance problems a Debian system was experiencing with SSH, the most widely used protocol for remotely logging in to devices over the Internet. Specifically, SSH logins were consuming too many CPU cycles and were generating errors with valgrind, a utility for monitoring computer memory.

    Through sheer luck and Freund’s careful eye, he eventually discovered the problems were the result of updates that had been made to xz Utils. On Friday, Freund took to the Open Source Security List to disclose the updates were the result of someone intentionally planting a backdoor in the compression software.

    It's hard to overstate the complexity of the social engineering and the inner workings of the backdoor. Thomas Roccia, a researcher at Microsoft, published a graphic on Mastodon that helps visualize the sprawling extent of the nearly successful endeavor to spread a backdoor with a reach that would have dwarfed the SolarWinds event from 2020.

    Enlarge

    What does the backdoor do?

    Malicious code added to xz Utils versions 5.6.0 and 5.6.1 modified the way the software functions. The backdoor manipulated sshd, the executable file used to make remote SSH connections. Anyone in possession of a predetermined encryption key could stash any code of their choice in an SSH login certificate, upload it, and execute it on the backdoored device. No one has actually seen code uploaded, so it's not known what code the attacker planned to run. In theory, the code could allow for just about anything, including stealing encryption keys or installing malware.

    Wait, how can a compression utility manipulate a process as security sensitive as SSH?

    Any library can tamper with the inner workings of any executable it is linked against. Often, the developer of the executable will establish a link to a library that's needed for it to work properly. OpenSSH, the most popular sshd implementation, doesn’t link the liblzma library, but Debian and many other Linux distributions add a patch to link sshd to systemd, a program that loads a variety of services during the system bootup. Systemd, in turn, links to liblzma, and this allows xz Utils to exert control over sshd.

    How did this backdoor come to be?

    It would appear that this backdoor was years in the making. In 2021, someone with the username JiaT75 made their first known commit to an open source project. In retrospect, the change to the libarchive project is suspicious, because it replaced the safe_fprint funcion with a variant that has long been recognized as less secure. No one noticed at the time.

    The following year, JiaT75 submitted a patch over the xz Utils mailing list, and, almost immediately, a never-before-seen participant named Jigar Kumar joined the discussion and argued that Lasse Collin, the longtime maintainer of xz Utils, hadn’t been updating the software often or fast enough. Kumar, with the support of Dennis Ens and several other people who had never had a presence on the list, pressured Collin to bring on an additional developer to maintain the project.

    In January 2023, JiaT75 made their first commit to xz Utils. In the months following, JiaT75, who used the name Jia Tan, became increasingly involved in xz Utils affairs. For instance, Tan replaced Collins' contact information with their own on oss-fuzz, a project that scans open source software for vulnerabilities that can be exploited. Tan also requested that oss-fuzz disable the ifunc function during testing, a change that prevented it from detecting the malicious changes Tan would soon make to xz Utils.

    In February of this year, Tan issued commits for versions 5.6.0 and 5.6.1 of xz Utils. The updates implemented the backdoor. In the following weeks, Tan or others appealed to developers of Ubuntu, Red Hat, and Debian to merge the updates into their OSes. Eventually, one of the two updates made its way into the following releases, according to security firm Tenable:

    Distribution Advisory Notes
    Fedora Rawhide https://www.redhat.com/en/blog/urgent-security-alert-fedora-41-and-rawhide-users Fedora Rawhide is the development distribution of Fedora Linux
    Fedora 41 https://www.redhat.com/en/blog/urgent-security-alert-fedora-41-and-rawhide-users
    Debian testing, unstable and experimental distributions versions 5.5.1alpha-0.1 to 5.6.1-1. https://lists.debian.org/debian-security-announce/2024/msg00057.html
    openSUSE Tumbleweed and openSUSE MicroOS https://news.opensuse.org/2024/03/29/xz-backdoor/ Backdoored version of xz was included in Tumbleweed and MicroOS between March 7 and March 28
    Kali Linux https://www.kali.org/blog/about-the-xz-backdoor/ Backdoored version of xz was included in Kali Linux (xz-utils 5.6.0-0.2) between March 26 and March 28

    There’s more about Tan and the timeline here.

    Can you say more about what this backdoor does?

    In a nutshell, it allows someone with the right private key to hijack sshd, the executable file responsible for making SSH connections, and from there to execute malicious commands. The backdoor is implemented through a five-stage loader that uses a series of simple but clever techniques to hide itself. It also provides the means for new payloads to be delivered without major changes being required.

    Multiple people who have reverse-engineered the updates have much more to say about the backdoor.

    Developer Sam James provided this overview:

    This backdoor has several components. At a high level:

    • The release tarballs upstream publishes don't have the same code that GitHub has. This is common in C projects so that downstream consumers don't need to remember how to run autotools and autoconf. The version of build-to-host.m4 in the release tarballs differs wildly from the upstream on GitHub.
    • There are crafted test files in the tests/ folder within the git repository too. These files are in the following commits:
      • tests/files/bad-3-corrupt_lzma2.xz (cf44e4b7f5dfdbf8c78aef377c10f71e274f63c0, 74b138d2a6529f2c07729d7c77b1725a8e8b16f1)
      • tests/files/good-large_compressed.lzma (cf44e4b7f5dfdbf8c78aef377c10f71e274f63c0, 74b138d2a6529f2c07729d7c77b1725a8e8b16f1)
    • A script called by build-to-host.m4 unpacks this malicious test data and uses it to modify the build process.
    • IFUNC, a mechanism in glibc that allows for indirect function calls, is used to perform runtime hooking/redirection of OpenSSH's authentication routines. IFUNC is a tool that is normally used for legitimate things, but in this case it is exploited for this attack path.

    Normally, upstream publishes release tarballs that are different than the automatically generated ones in GitHub. In these modified tarballs, a malicious version of build-to-host.m4 is included to execute a script during the build process.

    This script (at least in versions 5.6.0 and 5.6.1) checks for various conditions like the architecture of the machine. Here is a snippet of the malicious script that gets unpacked by build-to-host.m4 and an explanation of what it does:

    if ! (echo "$build" | grep -Eq "^x86_64" > /dev/null 2>&1) && (echo "$build" | grep -Eq "linux-gnu$" > /dev/null 2>&1);then
    • If amd64/x86_64 is the target of the build
    • And if the target uses the name linux-gnu (mostly checks for the use of glibc)

    It also checks for the toolchain being used:

    if test "x$GCC" != 'xyes' > /dev/null 2>&1;then
exit 0
fi
if test "x$CC" != 'xgcc' > /dev/null 2>&1;then
exit 0
fi
LDv=$LD" -v"
if ! $LDv 2>&1 | grep -qs 'GNU ld' > /dev/null 2>&1;then
exit 0

    And if you are trying to build a Debian or Red Hat package:

    if test -f "$srcdir/debian/rules" || test "x$RPM_ARCH" = "xx86_64";then

    This attack thusly seems to be targeted at amd64 systems running glibc using either Debian or Red Hat derived distributions. Other systems may be vulnerable at this time, but we don't know.

    In an online interview, developer and reverse-engineer HD Moore confirmed the Sam James suspicion that the backdoor targeted either Debian or Red Hat distributions.

    “The attack was sneaky in that it only did the final steps of the backdoor if you were building the library on amd64 (intel x86 64-bit) and were building a Debian or a RPM package (instead of using it for a local installation),” he wrote.

    Paraphrasing observations from researchers who collectively spent the weekend analyzing the malicious updates, he continued:

    When verifying an SSH public key, if the public key matches a certain fingerprint function, the key contents are decrypted using a pre-shared key before the public key is actually verified. The decrypted contents are then passed directly to system.

    If the fingerprint doesn't match or the decrypted contents don't match a certain format, it falls back to regular key verification and no-one's the wiser.

    The backdoor is super sneaky. It uses a little-known feature of the glibc to hook a function. It only triggers when the backdoored xz library gets loaded by a /usr/bin/sshd process on one of the affected distributions. There may be many other backdoors, but the one everyone is talking about uses the function indirection stuff to add the hook. The payload was encoded into fake xz test files and runs as a shellcode effectively, changing the SSH RSA key verification code so that a magic public key (sent during normal authentication) let the attacker gain access

    ​​Their grand scheme was:

    1) sneakily backdoor the release tarballs, but not the source code

    2) use sockpuppet accounts to convince the various Linux distributions to pull the latest version and package it

    3) once those distributions shipped it, they could take over any downstream user/company system/etc

    Researchers from networking firm Akamai also explain well how the backdoor works:

    The backdoor is quite complex. For starters, you won’t find it in the xz GitHub repository (which is currently disabled, but that’s besides the point). In what seems like an attempt to avoid detection, instead of pushing parts of the backdoor to the public git repository, the malicious maintainer only included it in source code tarball releases. This caused parts of the backdoor to remain relatively hidden, while still being used during the build process of dependent projects.

    The backdoor is composed of many parts introduced over multiple commits:

    • Using IFUNCs in the build process, which will be used to hijack the symbol resolve functions by the malware
    • Including an obfuscated shared object hidden in test files
    • Running a script set during the build process of the library that extracts the shared object (not included in the repository, only in releases, but added to .gitignore)
    • Disabling landlocking, which is a security feature to restrict process privileges

    The execution chain also consists of multiple stages:

    • The malicious script build-to-host.m4 is run during the library’s build process and decodes the “test” file bad-3-corrupt_lzma2.xz into a bash script
    • The bash script then performs a more complicated decode process on another “test” file, good-large_compressed.lzma, decoding it into another script
    • That script then extracts a shared object liblzma_la-crc64-fast.o, which is added to the compilation process of liblzma

    This process is admittedly hard to follow. We recommend Thomas Roccia’s infographic for a great visual reference and in-depth analysis.

    The shared object itself is compiled into liblzma, and replaces the regular function name resolution process. During (any) process loading, function names are resolved into actual pointers to the process memory, pointing at the binary code. The malicious library interferes with the function resolving process, so it could replace the function pointer for the OpenSSH function RSA_public_decrypt (Figure 1).

    It then points that function to a malicious one of its own, which according to research published by Filippo Valsorda, extracts a command from the authenticating client’s certificate (after verifying that it is the threat actor) and passes it on to the system() function for execution, thereby achieving RCE prior to authentication.

    The liblzma hooking process
    Enlarge / The liblzma hooking process
    Akamai

    What more do we know about Jia Tan?

    At the moment, extremely little, especially for someone entrusted to steward a piece of software as ubiquitous and as sensitive as xz Utils. This developer persona has touched dozens of other pieces of open source software in the past few years. At the moment, it’s unknown if there was ever a real-world person behind this username or if Jia Tan is a completely fabricated individual.

    Additional technical analysis is available from the above Bluesky thread from Valsorda, researcher Kevin Beaumont, and Freund’s Friday disclosure.

    Is there a CVE tracking designation?

    Yes, it's CVE-2024-3094.

    How do I know if the backdoor is present on my device?

    There are several ways. One is this page from security firm Binarly. The tool detects implementation of IFUNC and is based on behavioral analysis. It can automatically detect invariants in the event a similar backdoor is implanted elsewhere.

    There's also a project called xzbot. It provides the following:

    • honeypot: fake vulnerable server to detect exploit attempts
    • ed448 patch: patch liblzma.so to use our own ED448 public key
    • backdoor format: format of the backdoor payload
    • backdoor demo: cli to trigger the RCE assuming knowledge of the ED448 private key

    Ars Pro Subscriber

    View on a single page
    Article PDF

    reader comments

    195
    Dan Goodin Dan Goodin is Senior Security Editor at Ars Technica, where he oversees coverage of malware, computer espionage, botnets, hardware hacking, encryption, and passwords. In his spare time, he enjoys gardening, cooking, and following the independent music scene.

    Promoted Comments

    Christarp
    The whole world got lucky that one developer was determined enough to discover the cause of a minor performance regression. Just makes you wonder what else hasn't yet been discovered in our open source tooling out there.

    Really eye opening, thank you Andres!

    Edit: And of course, Dan for the wonderful article.
    April 1, 2024 at 7:16 am
    BarnSoftEng
    It should be noted that the attack only works because Debian and Redhat added functionality to sshd that is not present in it as distributed by its developers. The extra functionality adds systemd interaction, which requires libsystemd which requires liblzma, a component of the (compromised) xz package.

    One should be wary of distributions adding functionality. Often it increases the attack surface, not only because of the modifications/additions themselves, but also by adding dependencies.
    This conclusion ignores the huge value that the patches that all OS apply to packages provides. Without patches much upstream software will not build, will not run correctly.
    April 1, 2024 at 7:49 am
    pseudonomous
    It should be noted that the attack only works because Debian and Redhat added functionality to sshd that is not present in it as distributed by its developers. The extra functionality adds systemd interaction, which requires libsystemd which requires liblzma, a component of the (compromised) xz package.

    One should be wary of distributions adding functionality. Often it increases the attack surface, not only because of the modifications/additions themselves, but also by adding dependencies.

    While this is true, to a degree, it's also quite possible that if there were a lot of interesting targets out there that didn't patch sshd to use systemd and libxz, that given the level of access they had, the actor here could almost certainly found some other way to put an obfuscated backdoor into the code.

    I mean, certainly the OpenSSH project is obviously in the clear, but having gained commit access to a critical library, I don't think we could feel safe that "Jia Tan" couldn't have constructed a different backdoor even if they couldn't use the particular behavior Debian and Ubuntu added to their version of SSH.

    And I'm not sure anybody feels totally safe until somebody audits the code to make sure there's nothing else lurking in here that might be dangerous.
    April 1, 2024 at 7:50 am
    BarnSoftEng
    So a prime reason this became potentially exploitable is libsystemd in OpenSSH. Need I say more.
    The prime reason is a very well funded and capable attacker looked for a way in.
    if not xz or systemd then they would have attacked via the next candidate weak point.
    April 1, 2024 at 7:51 am
    om1
    The prime reason is a very well funded and capable attacker looked for a way in.
    if not xz or systemd then they would have attacked via the next candidate weak point.
    I do know that the OpenSSH devs are very strict on not increasing attack surface. The main reason why they did not want to link to systemd in the distributed version. You can be sure that OpenSSH as distributed contains less opportunity for attackers than the modified versions. That's my main point: modifications by distributions are not always improvements.
    April 1, 2024 at 8:20 am
    MikeGale
    We are lucky that this was detected and that some competent people have moved in to analyse. I presume more analysis is still being done. (Thanks to them.)

    What we don't know is how many other similar attacks have been deployed, and how many are in preparation.

    What can decent people do to reduce our risks?

    Thoughts that come to mind:
    1. Some of those who do this, often thankless, maintenance work might like more support. This might be financial contributions, or people (who know what they're doing) reviewing code submissions. Those who incorporate these libraries into their own programs (and all users) should maybe think about this. If there were a "donate to the maintainers" button on a story like this, that would convert the immediate story into something of greater value, if the maintainer would like that.
    2. Some of the maintainers might appreciate recognition. Some won't, but worth considering.
    3. Some who use the libraries can improve the checking they do.
    4. Unpleasant people who harass maintainers should be detected and treated appropriately.
    April 1, 2024 at 9:17 am
    TheMongoose
    "This developer persona has touched dozens of other pieces of open-source software in the past few years.". Well, I guess the Opensource community have some codes to review. Maybe the xz incident is only the tips of the iceberg.
    Right? I didn’t realise until I read this article just how far back this started. Now you’ve got to wonder: if this was a group, how many other so-far benign identities they have, contributing good code and building up reputations. And how easy it would be for them to slip a little something into another fundamental tool.
    April 1, 2024 at 9:18 am
    ← Previous story Next story →

    Related Stories

      Today on Ars

        • Store
        • Subscribe
        • About Us
        • RSS Feeds
        • View Mobile Site
        • Contact Us
        • Staff
        • Advertise with us
        • Reprints

        Newsletter Signup

        Join the Ars Orbital Transmission mailing list to get weekly updates delivered to your inbox. Sign me up →

        CNMN Collection
        WIRED Media Group
        © 2024 Condé Nast. All rights reserved. Use of and/or registration on any portion of this site constitutes acceptance of our User Agreement (updated 1/1/20) and Privacy Policy and Cookie Statement (updated 1/1/20) and Ars Technica Addendum (effective 8/21/2018). Ars may earn compensation on sales from links on this site. Read our affiliate link policy.
        Your California Privacy Rights | Do Not Sell My Personal Information
        The material on this site may not be reproduced, distributed, transmitted, cached or otherwise used, except with the prior written permission of Condé Nast.
        Ad Choices
