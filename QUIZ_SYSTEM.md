# Quiz Markdown System Documentation

## Overview

The curriculum uses a clean markdown syntax for quizzes that is automatically converted to interactive HTML during the build process. All existing quizzes have been converted to this new format.

## Quiz Syntax

Write quizzes using the `:::quiz{...}:::` format:

```markdown
:::quiz{correct: 1, explanation: "This explanation appears when users check their answer"}
What is the question text?

```python
# Optional code examples
x = 10
```

- Option A
- Option B (CORRECT)
- Option C
- Option D
:::
```

### Configuration Options

- **`correct`**: 0-based index of the correct answer (0 = first option, 1 = second option, etc.)
- **`explanation`**: Text shown after users check their answer
- **`(CORRECT)` marker**: Optional - indicates the correct option in the markdown

## Examples

### Simple True/False Question
```markdown
:::quiz{correct: 0, explanation: "Python uses zero-based indexing for all sequences."}
What is the index of the first item in a Python list?

- 0 (CORRECT)
- 1
- -1
:::
```

### Question with Code Block
```markdown
:::quiz{correct: 1, explanation: "The loop prints 0, 1, 2 because range(3) generates numbers from 0 up to (but not including) 3."}
What output does this code produce?

```python
for i in range(3):
    print(i)
```

- 1, 2, 3
- 0, 1, 2 (CORRECT)
- 0, 1, 2, 3
- 3
:::
```

### Multiple Choice with Complex Explanation
```markdown
:::quiz{correct: 2, explanation: "NumPy's equivalent of vectorized if-else is np.where() which efficiently applies the condition row-by-row."}
Which NumPy function creates a new column based on a conditional?

- np.if_else()
- np.conditional()
- np.where() (CORRECT)
- np.select()
:::
```

## Building the Book

The quiz markdown syntax must be converted to HTML before building. Use this process:

### Option 1: One Command
```bash
node process-quizzes.js && mdbook build
```

### Option 2: Two Steps
```bash
# Step 1: Convert all quiz markdown to HTML
node process-quizzes.js

# Step 2: Build the book
mdbook build
```

### What process-quizzes.js Does
- Scans all markdown files in `src/`
- Finds all `:::quiz{...}:::` blocks
- Converts them to interactive HTML quizzes
- Preserves all other markdown content
- Reports the number of files and quizzes processed

## Adding New Quizzes

1. **Write the quiz** in your markdown file using the `:::quiz{...}:::` syntax
2. **Run `node process-quizzes.js`** to convert markdown to HTML
3. **Run `mdbook build`** to build the book
4. **Open `book/index.html`** in a browser to test

## File Structure

```
curriculum-ds/
├── src/                              # Markdown source files (contain quiz syntax)
│   ├── 02-intro-python/
│   │   └── conditional-logic.md      # Quiz markdown here
│   └── ...
├── book/                             # Generated HTML output
├── theme/
│   ├── quiz.css                      # Quiz styling
│   ├── quiz.js                       # Quiz interactivity
│   └── ...
├── process-quizzes.js               # Process markdown quizzes → HTML
├── convert-qa-to-quiz.js            # Convert old Q&A format (if needed)
├── preprocessors/                   # Deprecated preprocessor (not in use)
└── book.toml                        # mdBook configuration
```

## Special Characters in Explanations

When using special characters in explanations, follow these rules:

- **Double quotes**: Use `\"` inside explanation string
- **Backslashes**: Use `\\` for literal backslashes  
- **Newlines**: Use `\n` inside explanation string

Example:
```markdown
:::quiz{correct: 0, explanation: "Use \\\"quotes\\\" with backslashes. Line breaks use \\n"}
Which special characters need escaping?

- Quotes, backslashes, and newlines (CORRECT)
- Only quotes
:::
```

## Converting Old Format Quizzes

If you have old-format quizzes (numbered Q&A format), convert them with:

```bash
node convert-qa-to-quiz.js
```

This converts all `#### **Question N:...**` style quizzes to the new `:::quiz{...}:::` syntax.

## Troubleshooting

### Quizzes visible but not interactive
- Ensure `theme/quiz.js` is in the `theme/` directory
- Check browser console for JavaScript errors
- Verify `quiz.js` loads when viewing an HTML page

### Quizzes not appearing at all
- Confirm `node process-quizzes.js` ran without errors
- Check that quiz markdown uses correct `:::quiz{...}:::` syntax
- Ensure the HTML conversion happened (check `src/` files)

### Build fails
- Verify `process-quizzes.js` completes successfully
- Check for unclosed `:::` delimiters in quiz blocks
- Ensure all explanation strings have matching quotes

### Git/Pre-commit Hook (Optional)

Add to `.git/hooks/pre-commit` to auto-process quizzes before commits:

```bash
#!/bin/bash
node process-quizzes.js
git add src/
```

## Tips for Quiz Authorship

1. **Keep options concise** - One or two short lines per option
2. **Make wrong options plausible** - Helps test understanding
3. **Write clear explanations** - Help students learn from mistakes
4. **Test code examples** - Run them to ensure they're correct
5. **Use parentheses sparingly** - Can confuse the parser
