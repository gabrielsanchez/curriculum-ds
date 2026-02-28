# Module Assessment

## Overview

Congrats! You've now completed the first module! You've learned what data science is and where it fits in the broader technology landscape, how to use Google Colab and Jupyter Notebooks to write and run code, how to save your work to GitHub and understand the commit workflow, and how to use AI coding assistants effectively and responsibly. This assessment asks you to demonstrate all of those skills by completing a structured notebook and submitting it to GitHub.

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/01-introduction-to-data-science/01-introduction-to-data-science-module-assessment_starter.ipynb).

## Learning Objectives

By the end of this assessment, you will have demonstrated the ability to:

- Demonstrate basic understanding of data science concepts, environment setup, and collaboration tools.
- Validate readiness to move on to Python fundamentals.

## Assessment Overview

This assessment has two parts:

1. **Coding Assessment** — a structured Colab notebook with four tasks to complete
2. **Knowledge Check** — three conceptual questions to answer in markdown cells

There is no Python data science code in this assessment — this module was about concepts and tools. What you're demonstrating is that your environment works, you understand the workflow, and you can think clearly about what you've learned.

## Coding Assessment

Complete the following tasks in a single Colab notebook. Each task includes instructions; complete them in order. When finished, save the notebook to your GitHub repository and submit the link.

---

### Task 1: Environment Verification

In a code cell, run the following and confirm all lines execute without error:

```python
import sys
import pandas as pd
import numpy as np
import matplotlib
import sklearn

print(f"Python:       {sys.version.split()[0]}")
print(f"pandas:       {pd.__version__}")
print(f"numpy:        {np.__version__}")
print(f"matplotlib:   {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print("\nEnvironment ready!")
```

**Expected output:** All five library versions printed without error.

---

### Task 2: Markdown Cell — Data Science in Your Own Words

Add a markdown cell and write a short explanation (3–5 sentences) that answers:
- What is data science, in your own words?
- Name one industry (e.g., healthcare, finance, retail) and describe one specific way data science is used there.
- What do you personally hope to build or analyze by the end of this curriculum?

Your answer should be original — do not copy the lesson text. Use the markdown cell to format your response with a heading and bullet points.

---

### Task 3: Python Basics Check

In a code cell, complete the following without using AI assistance for the actual answers (you may use it to check your work):

```python
# 1. Create a variable called 'my_name' and assign it your name as a string
my_name = ___

# 2. Create a variable called 'birth_year' and assign it your birth year as an integer
birth_year = ___

# 3. Calculate your approximate age and store it in 'my_age'
my_age = ___

# 4. Print a sentence using an f-string: "My name is X and I am approximately Y years old."
print(___)

# 5. Create a list called 'ds_tools' containing at least 4 tools from this module
ds_tools = ___

# 6. Print the number of tools in your list using len()
print(___)
```

This is not a trick — the goal is to verify that you can write basic Python expressions and that your notebook runs correctly.

---

### Task 4: GitHub Workflow

1. Save this completed notebook to your GitHub repository using **File → Save a copy in GitHub**
2. Use a descriptive commit message such as `"Complete Module 1 assessment"` (not `"update"` or `"save"`)
3. Navigate to your repository on GitHub and confirm the notebook appears there with your commit message
4. In a markdown cell in your notebook, paste the URL of the notebook file on GitHub (it will look like `https://github.com/yourname/repo/blob/main/module1_assessment.ipynb`)

---

### Task 5: Responsible AI Use

In a markdown cell, briefly answer (2–3 sentences each):

1. You're stuck on a data cleaning error and have been debugging for 20 minutes. How would you use an AI assistant to help, and what would you do after it gives you a response?

2. Your instructor assigns a graded notebook exercise. What is the appropriate way to use AI tools while completing it?

These are open-ended reflection questions — there are no trick answers. Thoughtful, honest responses earn full credit.

---

## Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Environment verification | 15 | All 5 libraries print correctly; no errors |
| Task 2: Data science in your own words | 25 | Original response; clear description; specific industry use case; personal goal stated |
| Task 3: Python basics | 25 | All 6 items complete; notebook runs without error; f-string and list used correctly |
| Task 4: GitHub workflow | 20 | Notebook saved to GitHub; descriptive commit message; GitHub URL pasted into notebook |
| Task 5: AI reflection | 15 | Thoughtful, specific responses demonstrating understanding of responsible AI use |
| **Total** | **100** | |

## Submission

1. Save your completed notebook to GitHub using **File → Save a copy in GitHub**
2. Copy the URL of the notebook file on GitHub
3. Submit the URL to the [AI Grader](https://ai-grader-pql9.onrender.com/)

## Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="Google Colab runs code in a virtual machine (runtime) that is terminated when you close the browser or after a period of inactivity. When you reopen the notebook, the file (containing your code and outputs) is intact — but the runtime is fresh with no variables in memory. The notebook must be re-executed from the top to restore the kernel state. This is why organizing your notebook so it runs correctly top-to-bottom is a professional habit.">
  <div class="quiz-question">
    <strong>Question 1:</strong> You close your Colab notebook and reopen it the next day. When you try to run a cell that uses a variable you defined yesterday, you get a `NameError`. What happened, and how do you fix it?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>The notebook file was corrupted — you need to recreate it from scratch.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Colab discarded the runtime when the session ended, clearing all variables from memory. To fix it, run all cells from the top of the notebook (`Runtime → Run all`).</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Variables expire after 24 hours in Colab due to a storage limitation.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The variable was accidentally deleted when the notebook was saved.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="Feature engineering is the practice of creating new input variables (features) by transforming or combining existing ones, in order to give machine learning algorithms better signal to learn from. Computing `price_per_sqft` from two raw columns is a classic example — the ratio captures something meaningful that neither column alone expresses. This happens during the preparation phase, before model training.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Which step of the data science lifecycle involves creating new variables from existing ones — such as computing `price_per_sqft` from `price` and `size`, or creating an `is_weekend` column from a date?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Data collection</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Model evaluation</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Feature engineering, which is part of the data preparation step</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Deployment</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="Machine learning models learn from data. If the training data was generated by biased human decisions — for example, hiring managers who historically favored candidates from certain schools — the model will learn to replicate that bias and apply it at scale, faster and more consistently than any human could. This is called **data bias** and is one of the most serious ethical challenges in applied AI. The fix is not a better algorithm; it is examining, auditing, and correcting the training data, and measuring model performance separately across demographic subgroups.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A data science team builds a model that predicts which job applicants to advance to interviews. After deployment, they discover the model consistently scores candidates from certain universities lower than equally qualified candidates from other schools. What is the most likely cause?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The model's algorithm is inherently biased against certain universities.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The team forgot to scale the features before training.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The training data — historical hiring decisions — reflected the biases of past human reviewers, and the model learned and automated those patterns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The model was overfit to the training data and generalized poorly.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

