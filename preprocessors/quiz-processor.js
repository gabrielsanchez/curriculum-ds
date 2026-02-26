#!/usr/bin/env node

/**
 * mdBook quiz preprocessor
 * Converts markdown quiz syntax to interactive HTML quizzes
 * 
 * Syntax:
 * :::quiz{correct: 1, explanation: "explanation text"}
 * Question text?
 * - Option A
 * - Option B (CORRECT)
 * - Option C
 * :::
 */

let input = '';

process.stdin.on('data', (chunk) => {
  input += chunk;
});

process.stdin.on('end', () => {
  try {
    let book;
    
    // Parse the input JSON
    if (!input || input.trim().length === 0) {
      // Empty input - create minimal structure
      book = { sections: [] };
    } else {
      book = JSON.parse(input);
    }
    
    // Ensure book has sections array
    if (!book || typeof book !== 'object') {
      book = { sections: [] };
    }
    
    // Process each section
    if (Array.isArray(book.sections)) {
      book.sections.forEach((section) => {
        // Process Chapter sections
        if (section && section.Chapter) {
          if (section.Chapter.content && typeof section.Chapter.content === 'string') {
            section.Chapter.content = processQuizzes(section.Chapter.content);
          }
        }
      });
    }
    
    // Output the modified book
    console.log(JSON.stringify(book));
  } catch (err) {
    // On parse error, try to output what we received
    process.stderr.write(`[quiz-processor] Parse error: ${err.message}\n`);
    // Try to send back a valid structure
    if (input) {
      try {
        const book = JSON.parse(input) || { sections: [] };
        console.log(JSON.stringify(book));
      } catch (e) {
        console.log(JSON.stringify({ sections: [] }));
      }
    } else {
      console.log(JSON.stringify({ sections: [] }));
    }
    process.exit(0);
  }
});

function processQuizzes(content) {
  if (!content || typeof content !== 'string') {
    return content;
  }
  
  let result = content;
  let quizNumber = 0;

  // Regular expression to match quiz blocks: :::quiz{config}...:::
  // Using a more careful approach to handle escaped quotes in config
  const quizRegex = /:::quiz\s*\{correct:\s*(\d+)\s*,\s*explanation:\s*"((?:[^"\\]|\\.)*?)"\}\s*\n([\s\S]*?):::(?:\n|$)/g;

  result = result.replace(quizRegex, (match, correct, explanation, quizContent) => {
    return convertQuizToHtml(quizContent.trim(), parseInt(correct), unescapeString(explanation), quizNumber++);
  });

  return result;
}

function unescapeString(str) {
  // Unescape characters that were escaped in the markdown
  return str
    .replace(/\\"/g, '"')
    .replace(/\\n/g, '\n')
    .replace(/\\\\/g, '\\');
}

function convertQuizToHtml(quizContent, correctIndex, explanation, quizIndex) {
  const lines = quizContent.split('\n');
  let questionText = '';
  let options = [];
  let subquestion = '';
  let inCodeBlock = false;
  let codeBlock = '';
  let i = 0;
  
  // Parse question (may be multiple lines including code blocks)
  while (i < lines.length) {
    const line = lines[i];
    
    // Handle code blocks
    if (line.trim().startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      if (inCodeBlock) {
        codeBlock += line + '\n';
      } else {
        codeBlock += line;
        subquestion = codeBlock;
        codeBlock = '';
      }
      i++;
      continue;
    }
    
    if (inCodeBlock) {
      codeBlock += line + '\n';
      i++;
      continue;
    }
    
    // Stop when we hit the first option
    if (line.trim().startsWith('-')) {
      break;
    }
    
    // Add non-empty lines to question
    if (line.trim()) {
      questionText += line + '\n';
    }
    
    i++;
  }
  
  questionText = questionText.trim();
  
  // Parse options
  while (i < lines.length) {
    const line = lines[i];
    
    if (line.trim().startsWith('-')) {
      let optionText = line.substring(line.indexOf('-') + 1).trim();
      // Remove (CORRECT) marker if present
      optionText = optionText.replace(/\s*\(CORRECT\)\s*$/, '');
      options.push(optionText);
    }
    
    i++;
  }
  
  // Generate HTML
  let html = `<div class="quiz-container" data-correct="${correctIndex}" data-explanation="${escapeHtml(explanation)}">\n`;
  html += `  <div class="quiz-question">\n`;
  html += `    <strong>Question ${quizIndex + 1}:</strong> ${questionText}\n`;
  
  if (subquestion) {
    html += `  </div>\n  <div class="quiz-subquestion">\n    ${subquestion}\n`;
  }
  
  html += `  </div>\n`;
  html += `  <div class="quiz-options">\n`;
  
  options.forEach((optionText, idx) => {
    html += `    <label class="quiz-option">\n`;
    html += `      <input type="radio" name="quiz-${quizIndex + 1}" value="${idx}">\n`;
    html += `      <label>${optionText}</label>\n`;
    html += `    </label>\n`;
  });
  
  html += `  </div>\n`;
  html += `  <button class="quiz-check-btn">Check Answer</button>\n`;
  html += `  <div class="quiz-feedback"></div>\n`;
  html += `</div>\n`;
  
  return html;
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, (char) => map[char]);
}
