# Pair Programming with AI

## Overview

In the previous lesson, you set up GitHub and learned how to save and version your notebooks. Now, before you write your first line of Python, you'll learn about one of the most powerful productivity tools available to modern data scientists: **AI coding assistants**. These tools can generate code, explain error messages, suggest improvements, and help you get unstuck — but only if you know how to use them effectively. In this lesson, you'll learn what AI assistants are good at, where they fall short, and how to integrate them into your learning and workflow without undermining your own skill development.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Learn how to use AI-driven coding assistants (Google Colab AI) effectively.
- Understand when to rely on AI vs. manual debugging and code reviews.

## Key Terms

**AI coding assistant:** A large language model (LLM) trained on code that can generate, explain, and debug code in response to natural language prompts. Examples: Google Colab AI, GitHub Copilot, Claude, ChatGPT.

**Prompt:** The input you provide to an AI assistant — a question, instruction, or code snippet that the model responds to.

**Context:** The surrounding information the AI uses to generate a response — your prompt, any code you share, and its training data. More context generally produces better responses.

**Hallucination:** When an AI model generates confident-sounding output that is factually incorrect or fabricated — for example, inventing a function that doesn't exist.

**Pair programming:** A development practice where two people work together on the same code — one "drives" (writes) while the other "navigates" (reviews and suggests). AI assistants act as a type of automated pair programmer.

**Debugging:** The process of finding and fixing errors in code.

**Autocomplete:** A feature that suggests code completions as you type, based on context. GitHub Copilot operates primarily as autocomplete; Colab AI can do both autocomplete and prompted generation.

## Introduction

AI coding assistants have fundamentally changed how developers and data scientists work. They are not replacements for understanding — they are amplifiers. A data scientist who understands what they're doing can use AI to work 2–3× faster. A learner who doesn't understand what they're doing and relies entirely on AI will produce code they can't explain, debug, or improve.

The key is knowing when to use AI as a tool versus when to struggle productively on your own. This lesson will help you draw that line.

## What AI Assistants Can Do Well

AI coding assistants are genuinely useful for:

**1. Boilerplate and repetitive code**
```python
# Prompt: "Write a function that loads a CSV, drops duplicate rows, and
# fills missing numeric values with the column median"

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df
```
This is accurate, fast, and saves time on code you know how to write but don't want to type.

**2. Explaining error messages**

Paste an error into the chat:
> "I got `KeyError: 'age'` when running `df['age'].fillna(0)`. What does this mean?"

The AI will explain that `'age'` is not a column in `df`, and suggest how to check which columns exist with `df.columns`.

**3. Explaining unfamiliar code**

```python
# Prompt: "Explain what this line does"
result = df.groupby("city")["price"].transform(lambda x: (x - x.mean()) / x.std())
```
The AI can explain: "This computes the z-score of `price` within each `city` group — it subtracts each city's mean price and divides by its standard deviation."

**4. Suggesting alternative approaches**
> "I'm using a for loop to apply a function to each row of a DataFrame. Is there a faster way?"

The AI will suggest `.apply()`, `.map()`, or vectorized operations, with examples.

**5. Documentation lookup**
> "What parameters does `pd.read_csv()` accept for handling missing values?"

Faster than searching documentation, and usually accurate for well-known libraries.

## Where AI Assistants Fall Short

Understanding the limitations is as important as understanding the capabilities:

**1. Hallucinations**

AI models sometimes invent functions, parameters, or behaviors that don't exist. Always run AI-generated code to verify it works — don't assume it's correct because it looks plausible.

```python
# An AI might generate something like:
df.fillna_smart(strategy="adaptive")  # This function does not exist
```

**2. Outdated knowledge**

AI models have a training cutoff. They may suggest deprecated APIs, old syntax, or libraries that have changed. For cutting-edge tools, check the official documentation.

**3. No access to your actual data**

An AI assistant doesn't see your DataFrame — it can only respond to what you describe. If your description is vague, the response will be generic. Share column names, data types, and sample rows in your prompt for better results.

**4. Can't run code**

Unless you're using a tool with code execution capability, the AI is generating text based on patterns — it can't verify that its code produces the right output on your data.

**5. Hidden bugs in complex logic**

AI is good at generating correct syntax. It's much less reliable at generating correct *logic* for complex, domain-specific problems. Always test outputs.

## Using Google Colab AI

Google Colab has a built-in AI assistant accessible in two ways:

**Inline generation:** Click the **✨ Generate** button above a code cell. A text box appears where you can describe what you want in plain English.

```
Prompt: "Create a bar chart showing the average salary by department,
         sorted from highest to lowest"
```

Colab AI generates the code directly into the cell. Run it, inspect the output, and adjust.

**Explain code:** Select code in a cell, right-click, and choose **Explain selection**. Colab AI explains the selected code in plain English — useful when encountering unfamiliar patterns.

**Fix error:** When a cell raises an error, Colab shows a "✨ Fix error" button. Clicking it passes the error and the code to the AI, which suggests a fix.

## Effective Prompting

The quality of AI output depends heavily on the quality of your prompt. A vague prompt gets a vague response.

| Weak prompt | Strong prompt |
|-------------|--------------|
| "Fix my code" | "This code raises a `ValueError: could not convert string to float`. The column `price` contains some values like `'N/A'`. How do I replace those with `NaN` before converting to float?" |
| "Make a chart" | "Create a seaborn boxplot showing the distribution of `fare` for each value of `pclass` in the Titanic DataFrame. Use a log scale on the y-axis." |
| "Explain groupby" | "Explain what `.groupby('city').agg({'price': 'mean', 'size': 'median'})` does, step by step" |

**Strategies for better prompts:**
- State the goal, not just the symptom: "I want to find the top 5 most common values" rather than "my code doesn't work"
- Include relevant context: column names, data types, what the DataFrame looks like
- Specify the libraries you're using: "using pandas" or "using matplotlib"
- Ask for explanations: "explain each line" ensures you understand what you're getting

## When to Use AI — and When Not To

```
New concept you're learning?
    → Try it yourself first. Struggle productively.
    → Use AI to check your work or ask for clarification after attempting.

Repetitive boilerplate?
    → Use AI freely. Your time is better spent elsewhere.

Stuck on a bug for more than 15 minutes?
    → Paste the error + relevant code into AI. It's often faster.

Something you already understand?
    → AI is fine to accelerate it.

An assessment or quiz?
    → Follow your institution's academic integrity policy.
    → Using AI to complete graded work you're supposed to do independently is academic dishonesty.
```

## The Learning Trap

The biggest risk with AI assistants for learners is the **illusion of understanding**. It's easy to:
1. Paste a prompt
2. Get working code
3. Run it
4. Move on

...without ever understanding what the code does. This feels productive but builds no skills. In an interview or on a job, you'll be asked to explain, debug, and extend code — and the AI won't be there.

**The productive loop looks like this:**
1. Attempt the problem yourself
2. Get stuck or produce something imperfect
3. Use AI to help — and read the explanation
4. Rewrite the solution in your own words / from scratch
5. Test it and make sure you understand every line

Following this loop, AI accelerates learning rather than replacing it.

## Conclusion

AI coding assistants are powerful tools — not magic, not cheating, and not a replacement for skill. Used well, they help you write boilerplate faster, understand error messages more quickly, and access documentation without leaving your notebook. Used poorly, they create an illusion of productivity while preventing real learning. The goal is to use AI to go further and faster, while maintaining the understanding needed to work independently. In the next lesson — the **Module Assessment** — you'll apply the conceptual and practical skills from this module to demonstrate readiness for Python fundamentals.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="AI-generated code can contain hallucinations, incorrect logic, or approaches that don&#039;t apply to your specific dataset. Running it without understanding it means you can&#039;t identify when it does something wrong — and it may silently produce incorrect results without raising an error. The correct workflow is: generate → explain → understand → verify. You should be able to explain every line of code you submit, regardless of whether AI helped you write it.">
  <div class="quiz-question">
    <strong>Question 1:</strong> You ask an AI assistant to "write code to clean my dataset." It generates a 30-line function using methods you don't recognize. What is the best next step?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Run the code immediately — if it produces no errors, it's correct.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Delete the code and try again with a different prompt.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Ask the AI to explain each line, then verify that the approach makes sense for your specific data before running it.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Copy the code directly into your final submission, since AI-generated code is always accurate.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="AI assistants respond to context. The third prompt provides: the library being used (pandas), the DataFrame name, the relevant columns and their types, the specific problem (12 missing values in `age`), the desired operation (median imputation), and an important constraint (use only the training set). This specificity allows the AI to generate targeted, correct code. Vague prompts produce generic, often useless responses.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Which of the following prompts is most likely to produce a useful, accurate AI response?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>"Fix my pandas code"</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>"Help with data cleaning"</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>"I have a pandas DataFrame called `df` with columns `age` (int), `income` (float), and `city` (str). The `age` column has 12 missing values. How do I fill them with the median age, using only the training set to compute the median?"</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>"Write data science code for me"</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Using AI to complete work without understanding it creates an illusion of competence. Grades reflect performance, not learning — and assessments in later modules build on concepts from earlier ones. In a technical interview or on the job, a data scientist is expected to write, read, debug, and explain code in real time without AI assistance. Someone who outsourced their learning to AI will hit a wall. The purpose of the curriculum is skill development, not just completion.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A classmate says: "I just use AI to write all my notebooks. It saves so much time and I always get full marks." What is the most significant risk of this approach?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>AI-generated code runs slower than human-written code.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The classmate may accumulate passing grades without developing the understanding needed to work independently — they won't be able to explain, debug, or extend the code in interviews, on the job, or in later, harder modules.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>GitHub can detect AI-generated code and flag it as plagiarism.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>AI assistants charge per notebook, making this approach expensive.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

