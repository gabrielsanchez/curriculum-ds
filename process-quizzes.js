#!/usr/bin/env node

/**
 * Process all markdown files to convert quiz syntax to HTML
 * Run this before mdbook build
 */

const fs = require('fs');
const path = require('path');

const srcDir = './src';
let filesProcessed = 0;
let quizzesConverted = 0;

function processQuizzes(content) {
  if (!content || typeof content !== 'string') {
    return content;
  }
  
  let result = content;
  let quizNumber = 0;

  // Regular expression to match quiz blocks
  const quizRegex = /:::quiz\s*\{correct:\s*(\d+)\s*,\s*explanation:\s*"((?:[^"\\]|\\.)*?)"\}\s*\n([\s\S]*?):::(?:\n|$)/g;

  result = result.replace(quizRegex, (match, correct, explanation, quizContent) => {
    quizzesConverted++;
    return convertQuizToHtml(quizContent.trim(), parseInt(correct), unescapeString(explanation), quizNumber++);
  });

  return result;
}

function unescapeString(str) {
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
  let codeBlockLines = [];
  let i = 0;
  
  // Parse question
  while (i < lines.length) {
    const line = lines[i];
    
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        // End of code block - join the lines and create subquestion
        subquestion = codeBlockLines.join('\n').trim();
        codeBlockLines = [];
      }
      inCodeBlock = !inCodeBlock;
      i++;
      continue;
    }
    
    if (inCodeBlock) {
      codeBlockLines.push(line);
      i++;
      continue;
    }
    
    if (line.trim().startsWith('-')) {
      break;
    }
    
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
    // Wrap code in <pre><code> for proper rendering
    html += `  </div>\n  <div class="quiz-subquestion">\n    <pre><code>${escapeHtml(subquestion)}</code></pre>\n`;
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

function walkDir(dir) {
  const files = fs.readdirSync(dir);
  
  files.forEach(file => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      walkDir(fullPath);
    } else if (file.endsWith('.md')) {
      try {
        let content = fs.readFileSync(fullPath, 'utf8');
        
        if (content.includes(':::quiz')) {
          const processed = processQuizzes(content);
          fs.writeFileSync(fullPath, processed, 'utf8');
          filesProcessed++;
        }
      } catch (err) {
        console.error(`Error processing ${fullPath}: ${err.message}`);
      }
    }
  });
}

// Main execution
console.log('Processing markdown files for quizzes...');
walkDir(srcDir);
console.log(`✓ Processed ${filesProcessed} files with ${quizzesConverted} quizzes converted`);
