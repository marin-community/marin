# Question
Title: Flammability (NFPA) - how is it defined?
I was reading up on a wonderful little chemical compound known as chlorine trifluoride ($\ce{ClF3}$). For a primer, check out Dr. Derek Lowe's blog post here: Sand Won't Save You This Time. The title of the post is quite telling; this compound is so reactive with other compounds - any other compound, really - that it will *set sand on fire*, making this old standby for extinguishing lab fires useless and putting the chemical in very rare company indeed. Obviously, water is a huge no-no for chlorine trifluoride; it will exothermically react with the water to produce clouds of steam containing two nasty acids (hydrochloric and hydrofluoric) and a host of chlorate and fluoride compounds. Somewhere in the middle of all the heat, fluorine and organic-ish compounds, you could probably even get some fluorine oxides, which would just LOVE to rapidly reduce their oxidation state (an event normally accompanied by shrapnel).

I was intrigued; I went and looked up the MSDS for this chemical to see what the instructions were for dealing with industrial spills of this stuff (apparently, there's a decent market for it in the semiconductor industry, and probably also for plastics to produce highly fluorinated polymers like Teflon). "A good pair of running shoes" can't be the only answer to a $\ce{ClF3}$ fire. *(well, actually, it pretty much is; get everyone in a one mile radius and two miles downwind the hell away from the area)*

Now, here's the question. While this stuff has some serious "synergy" when it comes to being an oxidizing agent (it's a better oxidizer than pure oxygen, loads better than pure chlorine, better even than pure fluorine gas), the NFPA flammability rating of the chemical is 0. This, despite reports from those who originally studied it that the chemical is hypergolic with every known fuel substance, and most things you wouldn't consider fuel (like asbestos, sand, concrete, brick, earth, test engineers). The resulting reaction certainly looks like fire; intense heat, bright light, sparks, smoke, the works. So, if chlorine trifluoride doesn't "burn" by the NFPA's definition, what *is* the NFPA's definition of "burning"?

# Answer
> 28 votes
You have found one of biggest flaws in the NFPA "Fire Diamond". The NFPA rates the flammability of the compound **as a fuel**. ClF<sub>3</sub> is not a fuel. It is an oxidant. ClF<sub>3</sub> is playing the role that oxygen normally does. It causes fuels (reductants) to burn by oxidizing them rapidly and exothermically. Oxidants are incapable of creating fire in absence of fuel regardless of temperature. Fires started by ClF<sub>3</sub> continue to burn after being smothered because oxygen isn't necessary to continue the oxidation process. And since ClF<sub>3</sub> is denser than air, it gets trapped under the sand. Many other powerful oxidants, like fluorine and perchloric acid, have low flammability ratings. Oxygen also has a flammability rating of zero. They do not burn; they cause other compounds to burn.

The scale for the flammability rating has to do with flash point, which is the temperature at which a fuel forms an ignitable mixture with air. Since ClF<sub>3</sub> doesn't burn (isn't a fuel), and doesn't react with air (N<sub>2</sub>, O<sub>2</sub>, CO<sub>2</sub>, Ar, etc, which are all inert towards oxidation), it does not have a measurable flash point. 

The danger posed by ClF<sub>3</sub> is summed in the yellow (reactivity or stability) diamond and the white (physical and chemical hazards) diamond. Your msds rates ClF<sub>3</sub> as a 3 in reactivity/stability, which is appropriate. Wikipedia's NPFA rating article, even mentions chlorine trifluoride:

> 3 - Capable of detonation or explosive decomposition but requires a strong initiating source, must be heated under confinement before initiation, reacts explosively with water, or will detonate if severely shocked (e.g. ammonium nitrate, chlorine trifluoride).

The white triangle will have appropriate hazard information. For ClF<sub>3</sub>, it has the symbols **OXY**, which means it is a strong oxidizer, and ~~W~~, which means it reacts with water. 

I have not found the specific reaction that ClF<sub>3</sub> undergoes with sand (mostly SiO<sub>2</sub>). I expect that it involves ${\ce{SiO2 -\> SiF}}\_6^{2-}$, except that this half reaction is not an oxidation.

---
Tags: redox, safety
---