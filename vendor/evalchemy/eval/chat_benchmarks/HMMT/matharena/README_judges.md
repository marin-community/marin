# Grading Model Outputs

As a judge, you will be grading one or more questions of a competition. Below, you can find detailed instructions on your task.

## Grading Scheme
First, you should create a grading scheme for your question. If you were assigned with another judge to the same question, discuss and create the grading scheme together to ensure that you both use the same. The final grading scheme should be created as a list of dictionaries in a json file, each dictionary containing specific steps for which points can be given. Specifically, each dictionary contains the following keys:
- `points` (float/int): Points associated with this step. Sum of all points for a single question should be 7.
- `title` (str): Very short description or title of the step. Should be unique across all dictionaries in this scheme.
- `desc` (str): Description of the step.

For instance, consider the example question:
```latex
Is it possible for an irrational number to an irrational power to be rational?
```
A good grading scheme for this question would be:
```json
[
    {
        "points": 1,
        "title": "Correct",
        "desc": "The answer should indicate that the answer to the question is 'Yes'"
    }, 
    {
        "points": 2,
        "title": "Example",
        "desc": "The answer gives a correct example of an irrational number to an irrational power that is rational."
    }, 
    {
        "points": 2,
        "title": "Irrationality",
        "desc": "The answer correctly argues that the found example consists of two irrational numbers."
    }, 
    {
        "points": 2,
        "title": "Rationality",
        "desc": "The answer correctly argues that the found example leads to a rational number."
    }
]
```

Pay attention to the following details when creating a grading scheme:
- Make sure that each step in the grading scheme is required for a correct proof. Note that there could exist different proofs for the same question.
- You can give partial credits for any given step. However, try to make the grading scheme as detailed as possible.

Once you have created your grading scheme, send it to the authors and clearly state for which question this grading scheme is.

## Grading
After the creation of a grading scheme, you will be asked to grade model solutions. For this purpose, you will be given anonymized pdfs. Thus, you will not know which model you are grading. Each PDF consists of several attempts by the same model for the question you are grading. For each pdf, you should produce another json file that contains your grade for each question, along with information for your reasoning. Specifically, your json file will consist of a list of dictionaries, each dictionary referring to one model answer. Make sure you list your gradings in the same order as given in the pdf. Each dictionary should consist of the following elements:
- `points` (float/int): Total number of points awarded to the model for the answer.
- `details` (list of dicts): A list of dictionaries, each dictionary referring to a specific point on the grading scheme. The keys for this dictionary should be
    - `title` (str): The title of the part you are grading in this specific element. Should be included as a title in the grading scheme.
    - `points` (float/int): The number of points awarded to the model for this part.
    - `desc` (str): Your reasoning behind the awarded points.

You should include all parts that were included in the grading scheme, even if the number of points for a specific part is 0. Referring once again to the example question, this is how your grades could look like when there are two attempts:

```json
[
    {
        "points": 5,
        "details": [
            {
                "title": "Correct",
                "points": 1,
                "desc": "The model correctly states that the answer to the question is 'Yes'."
            },
            {
                "title": "Example",
                "points": 2,
                "desc": "The example given by the model is correct."
            },
            {
                "title": "Irrationality",
                "points": 0,
                "desc": "The model forgets to argue about the irrationality of the real numbers in the example."
            },
            {
                "title": "Rationality",
                "points": 2,
                "desc": "The model correctly argues that the power of the two numbers is rational."
            }
        ]
    },
    {
        "points": 1,
        "details": [
            {
                "title": "Correct",
                "points": 1,
                "desc": "The model correctly states that the answer to the question is 'Yes'."
            },
            {
                "title": "Example",
                "points": 0,
                "desc": "The example given by the model is incorrect, as neither of the real numbers (2 and 2) is irrational."
            },
            {
                "title": "Irrationality",
                "points": 0,
                "desc": "The numbers are not irrational, therefore no points can be awarded for this part."
            },
            {
                "title": "Rationality",
                "points": 0,
                "desc": "While the model correctly argues that $2^2$ is rational, this is trivial and does not prove the required statement."
            }
        ]
    }
]
```

Once you created your json file, you can send it back to the organizers. Name the file using the unique id given to the PDF associated with the file.

To give you an idea how your grading will be displayed on the website, you can find a screenshot below of the example given above (in the case where the first and second grading were made by two judges for the same sample).

![Image displaying the grades by judges on the website](images/judge_overview.png)