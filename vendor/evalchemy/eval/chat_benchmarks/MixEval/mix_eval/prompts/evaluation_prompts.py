MULTI_CHOICE_PROMPT = "Answer with the option letter from the given choices directly."
FREE_FORM_PROMPT = "Answer the question shortly."
# FREE_FORM_PROMPT_QUAC = "Answer the question using a short excerpt (span) from the given text."
FREE_FORM_PROMPT_BBH = "Answer the question. \nLet's think step by step."
FREE_FORM_PROMPT_GSM8k = "Answer the question. \nLet's think step by step."
FREE_FORM_PROMPT_MATH = "Answer the question. \nLet's think step by step."

FIVE_SHOT_PREFIX_FREEFORM = """Question: The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume?
Answer the question. 
Let's think step by step.
Given:
The formula for the volume of a cone is $V = \frac{1}{3}Bh$
The area of the base (B) is 30 square units
The height (h) is 6.5 units
Step 1: Identify the values for B and h.
B = 30 square units
h = 6.5 units
Step 2: Substitute the values into the formula.
$V = \frac{1}{3} \times 30 \times 6.5$
Step 3: Multiply the values.
$V = \frac{1}{3} \times 195$
Step 4: Simplify the fraction.
$V = 65$ cubic units
So the final answer is: 65.


Question: As of the census of 2000, there were 7,791 people, 3,155 households, and 2,240 families residing in the county.  The population density was 16 people per square mile (6/km\u00b2).  There were 3,723 housing units at an average density of 8 per square mile (3/km\u00b2).  The racial makeup of the county was 97.63% Race (United States Census), 0.18% Race (United States Census) or Race (United States Census), 1.07% Race (United States Census), 0.19% Race (United States Census), 0.03% Race (United States Census), 0.22% from Race (United States Census), and 0.69% from two or more races.  1.26% of the population were Race (United States Census) or Race (United States Census) of any race. 33.7% were of germans, 13.9% swedish people, 10.1% irish people, 8.8% united states, 7.0% english people and 5.4% Danish people ancestry according to Census 2000.
Which group is smaller according to the census: households or families?
Answer the question shortly.
families


Question: Sort the following words alphabetically: List: behold oxalic maybe hew steel termcap pray stiffen dissipate misogyny format dew
Answer the question. 
Let's think step by step.
To sort the words alphabetically, we can follow these steps:
List the words as given.
Compare the first letter of each word.
If the first letters are the same, compare the second letters, and so on.
Arrange them in alphabetical order based on these comparisons.
Here are the words given:
behold
oxalic
maybe
hew
steel
termcap
pray
stiffen
dissipate
misogyny
format
dew
So the final answer is:
behold
dew
dissipate
format
hew
maybe
misogyny
oxalic
pray
steel
stiffen
termcap


Question: when was the last time ku won the championship
Answer the question shortly.
2022


Question: Hoping to rebound from their divisional loss to the Jaguars, the Texans stayed at home for a Week 4 duel with the Oakland Raiders.  Houston delivered the opening shot of the first quarter with kicker Kris Brown's 26-yard field goal, followed by the Raiders tying the game with a 46-yard field goal from kicker Sebastian Janikowski.  The Texans would take full command in the second quarter as Brown nailed a 34-yard field goal, followed by running back Steve Slaton getting a 32-yard touchdown run and catching an 18-yard touchdown pass from quarterback Matt Schaub.  Oakland would close out the half with Janikowski's 33-yard field goal.  In the third quarter, Houston would continue its domination with rookie linebacker Brian Cushing tackling Raiders running back Justin Fargas in his own endzone for a safety, immediately followed by wide receiver Jacoby Jones returning a kickoff 95 yards for a touchdown.
How many yards was the second longest field goal?
Answer the question shortly.
34


"""

FIVE_SHOT_PREFIX_MULTIPLECHOICE = """According to cell classification, prokaryotic cells are separated from eukaryotic cells. Which feature is often used to distinguish prokaryotic cells from eukaryotic cells?
A. life processes
B. size differences
C. plasma membranes
D. energy molecules
Answer with the option letter from the given choices directly.
B

As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective.
is elder scrolls online the same as skyrim
A. No
B. Yes
Answer with the option letter from the given choices directly.
A

connection
You can share files with someone if you have a connection to a what?
A. freeway
B. radio
C. wires
D. computer network
E. electrical circuit
Answer with the option letter from the given choices directly.
D

Approximately what percentage of participants in Milgram's obedience experiments thought they delivered the maximum amount of shock possible?
A. 0
B. 20
C. 40
D. 60
Answer with the option letter from the given choices directly.
D

How to check your Facebook feed
Which solution is correct?
A. Log in to Facebook. Click on the bell shaped button at the top right of your Facebook home window.
B. Log in to Facebook. Click on the bell shaped button at the top left of your Facebook home window.
Answer with the option letter from the given choices directly.
A

"""


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt_multichoice(entry):
    prompt = entry["prompt"]
    parsed_options = parse_options(entry["options"])
    if (
        "context" in entry
        and str(entry["context"]).lower() != "none"
        and str(entry["context"]).lower() != "null"
        and str(entry["context"]).replace(" ", "") != ""
    ):
        context = entry["context"]
        prompt = f"{context}\n{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"
    else:
        prompt = f"{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"
    return prompt


def construct_prompt_freeform(entry):
    prompt = entry["prompt"]
    if entry["benchmark_name"] == "QuAc":
        raise NotImplementedError("QuAC freeform prompt not implemented yet.")
    elif entry["benchmark_name"] == "BBH":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_BBH}"
    elif entry["benchmark_name"] == "GSM8k":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_GSM8k}"
    elif entry["benchmark_name"] == "MATH":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_MATH}"
    else:
        prompt = f"{prompt}\n{FREE_FORM_PROMPT}"
    if (
        "context" in entry
        and str(entry["context"]).lower() != "none"
        and str(entry["context"]).lower() != "null"
        and str(entry["context"]).replace(" ", "") != ""
    ):
        context = entry["context"]
        prompt = f"Question: {context}\n{prompt}"
    else:
        prompt = f"Question: {prompt}"
    return prompt


if __name__ == "__main__":
    # mp_input = {'context': "How to check your Facebook feed", 'prompt': "Which solution is correct?", 'options': ["Log in to Facebook. Click on the bell shaped button at the top right of your Facebook home window.", "Log in to Facebook. Click on the bell shaped button at the top left of your Facebook home window."]}
    ff_input = {
        "context": "According to some sources 363 civilians were killed in Kavadarci, 230 in Negotino and 40 in Vatasha.",
        "prompt": "What were the 3 villages that people were killed in?",
        "benchmark_name": "MATH",
    }

    # prompt_mp = construct_prompt_multichoice(mp_input)
    # print(prompt_mp)
    prompt_ff = construct_prompt_freeform(ff_input)
    print(prompt_ff)
