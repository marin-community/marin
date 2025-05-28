SlidePlayer
Upload
  • My presentations
  • Profile
  • Feedback
  • Log out

Log in

Auth with social network:

Forgot your password?

Download presentation

We think you have liked this presentation. If you wish to download it, please recommend it to your friends in any social system. Share buttons are a little bit lower. Thank you!

Buttons:

Presentation is loading. Please wait.

Presentation is loading. Please wait.

Congestion and crowding Games Pasquale Ambrosio* Vincenzo Bonifaci + Carmine Ventre* *University of Salerno + University “La Sapienza” Roma.

Published byNathalie Dimmick Modified over 10 years ago

Similar presentations


Presentation on theme: "Congestion and crowding Games Pasquale Ambrosio* Vincenzo Bonifaci + Carmine Ventre* *University of Salerno + University “La Sapienza” Roma."— Presentation transcript:

1 Congestion and crowding Games Pasquale Ambrosio* Vincenzo Bonifaci + Carmine Ventre* *University of Salerno + University “La Sapienza” Roma

2 What is a game? Classical example: BoS game Players Strategies: S 1 = S 2 = {Bach, Stravinsky} Payoff functions: u 1 (B,B) = 2, u 2 (B,S) = 0, … Equilibria: i.e. (B,B) and (S,S) are Nash equilibria (2,1)(0,0) (1,2) Bach Stravinsky player 1 player 2 A strategy s (e.g. (B,B)) is a Nash equilibrium iff for all players i it holds: u i (s -i,s) >= u i (s -i,s’) for each s’ in S i

3 (General) Congestion Games resources  roads 1,2,3,4 players  driver A, driver B strategies: which roads I use for reach my destination?  A wants to go in Salerno  e.g. S A ={{1,2},{3,4}}  B wants to go in Napoli  e.g. S B ={{1,4},{2,3}} what about the payoffs? Roma Salerno Milano Napoli road 1 road 2 road 3 road 4 AB

4 Payoffs in (G)CG: an example A choose path 1,2 B choose path 1,4 u A = - (c 1 (2) + c 2 (1)) = - 4 u B = - (c 1 (2) + c 4 (1)) = - 5 Roma Salerno Milano Napoli road 1 road 2 road 3 road 4 A B c 1 (1)=2c 1 (2)= 3 c 2 (1)=1c 2 (2)= 4 c 3 (1)=4c 3 (2)= 6 c 4 (1)=2c 4 (2)= 5 Costs for the roads SIRF (Small Index Road First) (-4,-5)(-6,-8) (-9,-7)(-8,-7) {1,2} {3,4} {1,4}{2,3} B A

5 Payoffs in (G)CG The payoff of i depends by congestion of the selected resources  u i is the opposite of the total congestion cost paid by i (C i ) For each resource there is a congestion cost (or delay) c k c k is function of n k (s) (the number of players that in the state s choose the resource k) Therefore the payoff of i in the state s is:

6 Congestion games: various models Symmetric CG  S i are all the same and payoffs are identical symmetric function of n-1 variables Single-choice CG  Each player can choose only one resource (anyone in the resources set E)  Unification of concepts of strategies and resources Subjective CG  Each player has a different experience of the congestion  As consequence every player has a specific payoff

7 Congestion games: various models (2) Network CG  Each player has a starting and terminal node and the strategies are the paths in the network Crowding game  Single-choice subjective CG with payoff non- increasing in n k (s) Weighted crowding game  Each player has a different weight upon the congestion

8 CGs: relations scheme

9 GCG and pure Nash equilibria Every game has at least one mixed Nash equilibrium A game with pure equilibria is “better” than another one with just mixed equilibria Thm (Rosenthal, 1973) Every (general) CG possesses at least one pure Nash equilibrium.

10 Rosenthal’s result The class of GCG is “nice” We know that there is a class of game for which it is possible to find a pure equilibrium (algorithm?) Introduce this (potential) function:

11 Potential functions A potential function can trace the “global payoff” of the system along the Nash dynamics Several kind of potential functions:  Ordinal potential function  Weighted potential function  (Exact) potential function  Generalized ordinal potential function

12 Ordinal potential function (2,1)(0,0) (1,2) u 1 (B,B) – u 1 (S,B) > 0 implies that P 1 (B,B) – P 1 (S,B) > 0 P 1 (B,B) – P 1 (S,B) > 0 implies that u 1 (B,B) – u 1 (S,B) > 0 and that u 2 (B,B) – u 2 (S,B) > 0 P 1 is an ordinal potential function for BoS game u 1 (B,S) – u 1 (S,S) < 0 implies that P 1 (B,S) – P 1 (S,S) < 0 A function P (from S to R) is an OPF for a game G if for every player i u i (s -i, x) - u i (s -i, z) > 0 iff P(s -i, x) - P(s -i, z) > 0 for every x, z in S i and for every s -i in S -i BoS = 40 02 P 1 =

13 Weighted potential function (1,1)(9,0) (0,9)(6,6) PD = 23/2 0 P 2 = u 1 (C,C) – u 1 (D,C) = 1 = 2 (P 2 (C,C) – P 2 (D,C)) u 2 (D,C) – u 2 (D,D) = 3 = 2 (P 2 (D,C) – P 2 (D,D)) C C D D u 1 (C,D) – u 1 (D,D) = 3 = 2 (P 2 (C,D) – P 2 (D,D)) u 2 (C,C) – u 2 (C,D) = 1 = 2 (P 2 (C,C) – P 2 (C,D)) P 2 is a (2,2)-potential function for PD game A function P (from S to R) is a w-PF for a game G if for every player i u i (s -i, x) - u i (s -i, z) = w i (P(s -i, x) - P(s -i, z)) for every x, z in S i and for every s -i in S -i G’ = (1,0)(2,4) (4,0)(3,1) 08 911 P 3 = u 1 (A,A) – u 1 (B,A) = 3 - 2 = 1/3 (P 3 (A,A) – P 3 (B,A)) u 2 (B,A) – u 2 (B,B) = 4 - 0 = 1/2 (P 3 (B,A) – P 3 (B,B)) A A B B u 1 (A,B) – u 1 (B,B) = 4 - 1 = 1/3 (P 3 (A,B) – P 3 (B,B)) u 2 (A,A) – u 2 (A,B) = 1 - 0 = 1/2 (P 3 (A,A) – P 3 (A,B)) P 3 is a (1/3,1/2)- potential function for the game G’

14 (Exact) potential function (1,1)(9,0) (0,9)(6,6) PD = 43 30 P 4 = u 1 (C,C) – u 1 (D,C) = P 4 (C,C) – P 4 (D,C) u 2 (D,C) – u 2 (D,D) = P 4 (D,C) – P 4 (D,D) C C D D u 1 (C,D) – u 1 (D,D) = P 4 (C,D) – P 4 (D,D) u 2 (C,C) – u 2 (C,D) = P 4 (C,C) – P 4 (C,D) P 4 is a potential function for PD game A function P (from S to R) is an (exact) PF for a game G if it is a w-potential function for G with w i = 1 for every i

15 Generalized ordinal potential function (1,0)(2,0) (0,1) G’’ = 03 12 P 5 = P 5 (A,B) – P 5 (A,A) > 0 implies that u 1 (A,B) – u 1 (A,A) > 0 but not that u 2 (A,B) – u 2 (A,A) > 0 A function P (from S to R) is an GOPF for a game G if for every player i u i (s -i, x) - u i (s -i, z) > 0 implies P(s -i, x) - P(s -i, z) > 0 for every x, z in S i and for every s -i in S -i P 5 is a generalized ordinal potential function for the game G’’ P 5 is not an ordinal potential function for the game G’’ A A B B

16 Potential games A game that admits an OPF is called an ordinal potential game A game that admits a weighted PF is called a weighted potential game A game that admits a PF is called a potential game Using the potential functions properties we obtain several interesting results  E.g., in such games find an equilibrium is equivalent to maximize the potential

17 Equilibria in Potential Games Thm (MS96) Let G be an ordinal potential game (P is an OPF). A strategy profile s in S is a pure equilibrium point for G iff for every player i it holds P(s) >= P(s -i, x) for every x in S i Therefore, if P has maximal value in S, then G has a pure Nash equilibrium. Corollary Every finite OP game has a pure Nash equilibrium.

18 Nash equilibrium P 4 maximal value An example (1,1)(9,0) (0,9)(6,6) PD = C C D D 43 30 P 4 = (4,4)(3,3) (0,0) PD(P 4 ) = C C D D Thm (MS96)

19 FIP: an important property A path in S is a sequence of states s.t. between every consecutive pair of states there is only one deviator A path is an improvement path w.r.t. G if each deviator has a sharp advantage moving u i (s k ) > u i (s k-1 ) G has the FIP if every improvement path is finite Clearly if G has the FIP then G has at least a pure equilibrium  Every improvement path terminates in an equilibrium point

20 FIP: an important property (2) Lemma Every finite OP game has the FIP. The converse is true? Lemma Let G be a finite game. Then, G has the FIP iff G has a generalized ordinal potential function. (1,0)(2,0) (0,1) G’’ = A A B B G’’ has the FIP ((B,A) is an equilibrium) any OPF must satisfies the following impossible relations: P(A,A) < P(B,A) < P(B,B) < P(A,B) = P(A,A) “No”

21 Congestion vs Potential Games Rosenthal states that Congestion games always admit pure Nash equilibria MS96’s work shows that potential games always admit pure Nash equilibria What is the relation? Thm Every congestion game is a potential game. Thm Every finite potential game is isomorphic to a congestion game.


Download ppt "Congestion and crowding Games Pasquale Ambrosio* Vincenzo Bonifaci + Carmine Ventre* *University of Salerno + University “La Sapienza” Roma."

Similar presentations


The role of compatibility in the diffusion of technologies in social networks Mohammad Mahdian Yahoo! Research Joint work with N. Immorlica, J. Kleinberg,

The role of compatibility in the diffusion of technologies in social networks Mohammad Mahdian Yahoo! Research Joint work with N. Immorlica, J. Kleinberg,

An Introduction to Game Theory Part V: Extensive Games with Perfect Information Bernhard Nebel.

An Introduction to Game Theory Part V: Extensive Games with Perfect Information Bernhard Nebel.

Inefficiency of equilibria, and potential games Computational game theory Spring 2008 Michal Feldman TexPoint fonts used in EMF. Read the TexPoint manual.

Inefficiency of equilibria, and potential games Computational game theory Spring 2008 Michal Feldman TexPoint fonts used in EMF. Read the TexPoint manual.

Some Problems from Chapt 13

Some Problems from Chapt 13

6.896: Topics in Algorithmic Game Theory Lecture 20 Yang Cai.

6.896: Topics in Algorithmic Game Theory Lecture 20 Yang Cai.

Price Of Anarchy: Routing

Price Of Anarchy: Routing

Game Theory Assignment For all of these games, P1 chooses between the columns, and P2 chooses between the rows.

Game Theory Assignment For all of these games, P1 chooses between the columns, and P2 chooses between the rows.

This Segment: Computational game theory Lecture 1: Game representations, solution concepts and complexity Tuomas Sandholm Computer Science Department Carnegie.

This Segment: Computational game theory Lecture 1: Game representations, solution concepts and complexity Tuomas Sandholm Computer Science Department Carnegie.

Bilinear Games: Polynomial Time Algorithms for Rank Based Subclasses Ruta Mehta Indian Institute of Technology, Bombay Joint work with Jugal Garg and Albert.

Bilinear Games: Polynomial Time Algorithms for Rank Based Subclasses Ruta Mehta Indian Institute of Technology, Bombay Joint work with Jugal Garg and Albert.

6.896: Topics in Algorithmic Game Theory Lecture 11 Constantinos Daskalakis.

6.896: Topics in Algorithmic Game Theory Lecture 11 Constantinos Daskalakis.

Congestion Games with Player- Specific Payoff Functions Igal Milchtaich, Department of Mathematics, The Hebrew University of Jerusalem, 1993 Presentation.

Congestion Games with Player- Specific Payoff Functions Igal Milchtaich, Department of Mathematics, The Hebrew University of Jerusalem, 1993 Presentation.

ECO290E: Game Theory Lecture 5 Mixed Strategy Equilibrium.

ECO290E: Game Theory Lecture 5 Mixed Strategy Equilibrium.

For any player i, a strategy weakly dominates another strategy if (With at least one S -i that gives a strict inequality) strictly dominates if where.

For any player i, a strategy weakly dominates another strategy if (With at least one S -i that gives a strict inequality) strictly dominates if where.

EC3224 Autumn Lecture #04 Mixed-Strategy Equilibrium

EC3224 Autumn Lecture #04 Mixed-Strategy Equilibrium

EC941 - Game Theory Lecture 7 Prof. Francesco Squintani

EC941 - Game Theory Lecture 7 Prof. Francesco Squintani

How Bad is Selfish Routing? By Tim Roughgarden Eva Tardos Presented by Alex Kogan.

How Bad is Selfish Routing? By Tim Roughgarden Eva Tardos Presented by Alex Kogan.

Noam Nisan, Michael Schapira, Gregory Valiant, and Aviv Zohar.

Noam Nisan, Michael Schapira, Gregory Valiant, and Aviv Zohar.

Game-theoretic analysis tools Necessary for building nonmanipulable automated negotiation systems.

Game-theoretic analysis tools Necessary for building nonmanipulable automated negotiation systems.

Game Theory Lecture 8.

Game Theory Lecture 8.

1 Algorithmic Game Theoretic Perspectives in Networking Dr. Liane Lewin-Eytan.

1 Algorithmic Game Theoretic Perspectives in Networking Dr. Liane Lewin-Eytan.

Similar presentations


About project
SlidePlayer
Terms of Service
Do Not Sell
My Personal
Information
Feedback
Privacy Policy
Feedback

© 2025 SlidePlayer.com Inc.
All rights reserved.

To make this website work, we log user data and share it with processors. To use this website, you must agree to our Privacy Policy, including cookie policy.     
Ads by Google
