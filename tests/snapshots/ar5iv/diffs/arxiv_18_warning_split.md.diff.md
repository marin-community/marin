Daylight quantum key distribution over 1.6 km
=============================================

Abstract
--------

Quantum key distribution (QKD) has been demonstrated over a
point-to-point $`1.6`$-km atmospheric optical path in full daylight.
This record transmission distance brings QKD a step closer to
surface-to-satellite and other long-distance applications.

Quantum cryptography was introduced in the mid-1980s as a new method for
generating the shared, secret random number sequences, known as
cryptographic keys, that are used in crypto-systems to provide
communications security (for a review see ). The appeal of quantum
cryptography (or more accurately, quantum key distribution, QKD) is that
its security is based on laws of nature and information-theoretically
secure techniques, in contrast to existing methods of key distribution
that derive their security from the perceived intractability of certain
problems in number theory, or from the physical security of the
distribution process.

Several groups have demonstrated QKD over multi-kilometer distances of
optical fiber , but there are many key distribution problems for which
QKD over line-of-sight atmospheric paths would be advantageous (for
example, it is impractical to send a courier to a satellite). Free-space
QKD was first demonstrated in 1990 over a point-to-point 32-cm table top
optical path, and recent work has produced atmospheric transmission
distances of 75 m (daytime) and 1 km (nighttime) over outdoor folded
paths (to a mirror and back). The close collocation of the QKD
transmitter and receiver in folded-path experiments is not
representative of practical applications and can result in some
compensation of turbulence effects. We have recently performed the first
point-to-point atmospheric QKD in full daylight, achieving a 0.5-km
transmission range , and here we report a record 1.6-km point-to-point
transmission in daylight, with a novel QKD system that has no active
polarization switching elements.

The success of QKD over atmospheric optical paths depends on the
transmission and detection of single-photons against a high background
through a turbulent medium. Although this problem is difficult, a
combination of temporal, spectral and spatial filtering can render the
transmission and detection problems tractable . The essentially
non-birefringent nature of the atmosphere at optical wavelengths allows
the faithful transmission of the single-photon polarization states used
in the free-space QKD protocol.

A QKD procedure starts with the sender, "Alice," generating a secret
random binary number sequence. For each bit in the sequence, Alice
prepares and transmits a single photon to the recipient, "Bob," who
measures each arriving photon and attempts to identify the bit value
Alice has transmitted. Alice's photon state preparations and Bob's
measurements are chosen from sets of non-orthogonal possibilities. For
example, using the B92 protocol Alice agrees with Bob (through public
discussion) that she will transmit a $`45^{}`$ polarized photon state
$`|45`$, for each "0" in her sequence, and a vertical polarized photon
state $`|v`$, for each "1" in her sequence. Bob agrees with Alice to
randomly test the polarization of each arriving photon with $`45^{}`$
polarization, $`|45`$, to reveal "1s," or horizontal polarization,
$`|h`$, to reveal "0s." In this scheme Bob will never detect a photon
for which he and Alice have used a preparation/measurement pair that
corresponds to different bit values, such as $`|h`$ and $`|v`$, which
happens for 50% of the bits in Alice's sequence. However, for the other
50% of Alice's bits the preparation and measurement protocol uses
non-orthogonal states, such as for $`|45`$ and $`|h`$, resulting in a
50% detection probability for Bob. Thus, by detecting single-photons Bob
identifies a random 25% portion of the bits in Alice's random bit
sequence, assuming a single-photon Fock state with no bit loss in
transmission or detection. This 25% efficiency factor, $`\eta _Q`$, is
the price that Alice and Bob must pay for secrecy.

Bob and Alice reconcile their common bits by revealing the locations,
but not the bit values, in the sequence where Bob detected photons;
Alice retains only those detected bits from her initial sequence. In
practical systems the resulting sifted key sequences , will contain
errors; a pure key is distilled from them using classical error
detection techniques. The single-photon nature of the transmissions
ensures that an eavesdropper, "Eve," can neither "tap" the key
transmissions with a beam splitter (BS), owing to the indivisibility of
a photon , nor faithfully copy them, owing to the quantum "no-cloning"
theorem . Furthermore, the non-orthogonal nature of the quantum states
ensures that if Eve makes her own measurements she will be detected
through the elevated error rate she causes by the irreversible "collapse
of the wavefunction" . From the observed error rate and a model for
Eve's eavesdropping strategy, Alice and Bob can calculate a rigorous
upper bound on the infomation Eve might have obtained. Then, using the
technique of generalized privacy amplification by public discussion
Alice and Bob can distill a shorter, final key on which Eve has less
than one bit of information.

The QKD transmitter ("Alice") in our experiment (Fig. 1)operates at a
clock rate $`R_0=1`$-MHz. On each "tick" of the clock a $`1`$-ns
vertically-polarized optical "bright pulse" is produced from a
"timing-pulse" diode laser whose wavelength is temperature controlled to
$`768`$ nm. After a $`100`$-ns delay one of two temperature-controlled
dim pulse "data" diode lasers emits a $`1`$-ns optical pulse that is
attenuated to the single-photon level and constrained by an interference
filter (IF) to $`773\pm 0.5`$ nm to remove wavelength information.
Polarizers set one data laser's output to be $`45^{}`$ polarized and the
other to be vertically polarized as required for the B92 protocol. The
choice of which data laser fires is determined by a random bit value
that is obtained by discriminating electrical noise. The random bit
value is indexed by the clock tick and recorded in Alice's computer
control system's memory. All three optical pulse paths are combined with
beamsplitters (BSs) into a single-mode (SM) optical fiber to remove
spatial mode information, and transmitted toward Bob's receiver through
a $`27\times `$ beam expander (to extend the system Rayleigh range). A
single-photon detector (SPD) located behind a matched IF in one of the
BS output ports is used to monitor the average photon number
$`\overline{n}`$ of the dim-pulses as follows: (1) a calibration
photon-number measurement is made from the rate at which a calibrated
single-photon counting module (SPCM) fires at the transmitter's SM
transmission-fiber output with a given input, (2) next the transmitter's
SPD count rate is calibrated to the SPCM firing rate with the same input
to determine the SPD efficiency, which is then (3) used with the
experimental SPD count rates to measure the transmitted $`\overline{n}`$
in key generation mode.

At the QKD receiver ("Bob") light pulses are collected by a $`8.9`$-cm
diameter Cassegrain telescope and directed into a polarization analysis
and detection system (Fig. 2). A bright pulse triggers a "warm"
avalanche photodiode (APD), which sets up a narrow $`5`$-ns coincidence
gate in which to test a subsequent dim pulse's polarization . A BS
randomly directs dim pulses along one of two paths. Polarization
elements along the upper path are set to transmit $`45^{}`$ polarization
in accordance with Bob's B92 "1" value, while along the lower path a
measurement for $`|h`$ to reveal "0"s is made using a polarizing
beamsplitter (PBS). (The PBS transmits $`|h`$ but reflects $`|v`$.) Each
analysis path contains a matched IF and couples to a SPD via multi-mode
(MM) fiber that provides limited spatial filtering, giving the receiver
a restricted $`200`$ $`\mu `$-radian field of view. For events on which
one of the two SPDs triggers during the coincidence gate, Bob can assign
a bit value to Alice's transmitted bit; upper-path SPD firings identify
"1"s, and lower-path SPD firings identify "0"s. He records these
detected bits in the memory of his computer control system, indexed by
the "bright pulse" clock tick. Bit generation is completed when Bob
communicates the locations, but not values, of his photon detections in
Alice's random bit-sequence over a public channel: wireless ethernet in
our experiment.

The QKD system was operated over a $`1.6`$-km outdoor range with
excellent atmospheric conditions on Friday $`13`$ August $`1999`$
beginning at 09:30 LST under cloudless New Mexico skies. By 11:30 LST
turbulence induced beam-spreading hindered our ability to efficiently
acquire data at low bit-error rates ($`BER`$), $`ϵ`$ (where $`BER`$,
$`ϵ`$, is defined as the ratio of the number of bits received in error
to the total number of bits received). The system efficiency,
$`\eta _{system}`$, which accounts for losses between the transmitter
and MM fibers at the receiver, and the receiver's SPDs efficiencies had
an average value of $`\eta _{system}0.13`$ with a standard deviation of
$`\sigma =0.04`$. Fluctuations in $`\eta _{system}`$ were caused by
turbulence induced beam-spreading and beam-wander; the typical
beam-wander was observed to be on the order of $`3`$ to $`5`$
$`\mu `$-radians. (Our present system has no beam-steering or
adaptive-optics technology to compensate for turbulence-induced
effects.) The $`\eta _Q=0.25`$ quantum efficiency of the B92 protocol
lowers the overall efficiency to $`\eta =\eta _Q\eta _{system}0.0325`$
and leads to a detection probability for Bob of
$`P_B=1exp(\eta \overline{n})`$. This gave a bit-rate of $`R5.4`$,
$`12.2`$, and $`17`$-kHz at $`\overline{n}0.2`$, $`0.35`$, and
$`0.5`$-photons per dim-pulse, respectively, when the lasers were pulsed
at $`R_0=1`$ MHz. Bits were transmitted in $`25`$, $`50`$, and $`100`$
k-bit blocks. A total of $`1.55`$ M-bits were sent in $`40`$ data
exchanges between Alice and Bob and $`17,420`$ bits of sifted key were
received. Table I includes a typical $`250`$-bit sample from one of
several $`1.6`$-km daylight transmissions on $`13`$ August $`1999`$. The
sifted key shown contains eight bit-errors (in bold) corresponding to
$`ϵ=3.2`$% for these $`250`$ bits and has a $`60`$:$`40`$ bias toward
ones (the average bias for all experiments on $`13`$ August $`1999`$ was
$`50.3`$:$`49.7`$ toward ones). The average $`BER`$ on all key material
acquired during the daylight transmissions was $`ϵ=5.3`$%. These
$`BER`$s would be regarded as unacceptably high in any conventional
telecommunications application but are tolerated in QKD because of the
secrecy of the bits.

The dominant $`BER`$ component is from the ambient solar background,
with a measured noise probability for both detectors of about
$`6.7\times 10^4`$ per coincidence gate, contributing about $`5.9`$% to
the $`ϵ=7.8`$% at $`\overline{n}=0.2`$ data, about $`2.4`$% to the
$`ϵ=4.1`$% at $`\overline{n}=0.35`$, and about $`1.9`$% to the
$`ϵ=4.1`$% at $`\overline{n}=0.5`$. (The ambient-background is somewhat
less than that expected from the daylight radiance , which we attribute
to Bob viewing the dark interior of the tent housing Alice's
transmitter.) Imperfections and misalignments of the polarizing elements
were the next largest contribution (about $`1.9`$%) to the total
$`BER`$s on 13 August 1999. Experience from previous experiments
suggests that this component of $`BER`$ can be reduced to about
$`0.5`$%. Detector dark noise ($`1,400`$ dark-counts per second) makes
an even smaller contribution of $`<0.1`$% to the $`BER`$. The dual-fire
rate --- the probability that both SPDs fire during a coincidence
window--- was $`0.0003`$, $`0.0007`$, and $`0.001`$ at
$`\overline{n}0.2`$, $`0.35`$, and $`0.5`$, respectively.

Alice and Bob can correct errors by transmitting error correction
information over the public channel, amounting to

$$f(ϵ)=ϵ\mathrm{log}_2ϵ(1ϵ)\mathrm{log}_2(1ϵ)$$ (1)\
bits per bit of sifted key in the Shannon limit. For example, for
$`ϵ=4.1`$%, $`f(0.041)=0.246`$. Practical error-correcting codes do not
achieve the Shannon limit, although the interactive scheme known as
Cascade , comes within about $`1.16f(ϵ)`$ for error rates up to $`5`$% .
Our experiments use a combination of block-parity checks and Hamming
codes achieving an efficiency equivalent to the $`Cascade`$ scheme but
with greater computational efficiency. The error correction information
is transmitted over the public channel and thus could provide
information about the key material to Eve, reducing Alice and Bob's
secret bit yield. (Alice and Bob could encrypt the error correction
information to deny Eve access to it, but at the cost of an equal number
of shared secret key bits .)

Alice and Bob now use "privacy amplification" to reduce any partial
knowledge gained by an eavesdropper to less than 1-bit of information.
(For discussions of eavesdropping strategies see References .) We have
not implemented privacy amplification at this time, but to estimate the
secret-key rate for our experiment and its dependencies on relative
parameters, we assume Eve is restricted to performing the combination of
the intercept-resend and beamsplitting attacks considered in . In this
case Alice and Bob could use the parities of random subsequences of
their error-corrected keys as their final secret key bits, resulting in
a compression to

$$F(ϵ)=(1\overline{n})2\sqrt{2}ϵ$$ (2)\
bits per bit of error-corrected key. The first term in Eq. 2 accounts
for the multi-photon fraction of Alice's dim-pulses, which are
susceptible to beamsplitting, while the second accounts for Eve
performing intercept-resend on a fraction of the pulses. The final
secret bit yield is therefore a fraction $`F(ϵ)f(ϵ)`$ the length of the
original sifted key. For $`\overline{n}0.05`$, under the conditions of
our $`13`$ August $`1999`$ experiment with $`\eta _{system}=0.13`$,
there is no net secret bit yield because of the large value $`f(ϵ)`$.
With increasing $`\overline{n}`$ the $`BER`$ decreases so rapidly that
the increased privacy amplification cost to protect against
beamsplitting is more than offset by the reduced error-correction cost,
and so the secret bit yield initially increases. However, for larger
$`\overline{n}`$ values, the privacy amplification factor $`F(ϵ)`$
required to compensate for beamsplitting of multi-photon pulses becomes
small, and the secret bit yield decreases. For $`\eta _{system}=0.13`$,
we find that the optimum $`\overline{n}`$ for our $`13`$ August $`1999`$
experiment is $`0.4`$ giving a secret bit yield of $`38.5`$% of the
sifted key length, and $`0.4`$% of the length of the transmitted
sequence. (With $`Cascade`$ the optimal $`\overline{n}`$ would also be
$`0.4`$ and the secret bit yield would be $`24.7`$% of the sifted key or
$`0.32`$% the length of the transmitted sequence, giving a secret bit
rate of $`3`$-kHz.) For smaller $`\eta _{system}`$ values under $`13`$
August $`1999`$ conditions the optimal $`\overline{n}`$ values are as
above but the secret bit yield is smaller; for $`\eta _{system}<0.04`$
there is no secret bit yield. (To protect against the attacks proposed
in , should they become feasible, we would need to reduce our background
further with a shorter coincidence gate window and narrower spectral
filters to have a non-zero secret-bit yield at the $`\overline{n}`$
values required.)

This paper reports QKD between a transmitter and receiver separated by a
$`1.6`$-km daylight atmospheric optical path. This transmission
distance, which was only limited by the length of the available range,
is the longest to date. Our system has no active polarization elements,
resulting in greater simplicity and security over previous experiments.
Secret bit rates of several kilo-Hertz protected against simple
beamsplitting and intersept-resend attacks have been shown to be
feasible. Such rates would enable the rekeying of cryptographic systems.
The system could be easily adapted to the BB84 four-state QKD protocol
or to use single-photon light sources once they are available, providing
protection against more sophisticated future attacks . Our results are
representative of practical situations showing that QKD could be used in
conjunction with optical communication systems and providing further
evidence for the feasibility of surface-to-satellite QKD . The
$`1.6`$-km optical path is similar in optical depth to the effective
turbulent atmospheric thickness encountered in a surface-to-satellite
application. Significant amounts of key-material (about $`15`$ k-bits)
with low $`BER`$s ($`ϵ3.0`$%) at low $`\overline{n}`$
($`\overline{n}0.2`$) were also taken at night and during light rain
over this $`1.6`$-km distance. Finally, we note that the variability of
system efficiency and background is a feature of atmospheric QKD that is
quite different from optical fiber systems.

We wish to thank P. G. Kwiat for QKD discussions, D. Derkacs for
technical support, and G. H. Nickel, D. Simpson, and E. Twyeffort for
discussions regarding bit-error detection protocols and privacy
amplification.
