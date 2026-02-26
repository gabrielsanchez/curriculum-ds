#!/usr/bin/env node

/**
 * Convert HTML quiz format to markdown format
 * Usage: node convert-quizzes.js [file.md]
 */

const fs = require('fs');
const path = require('path');

function convertQuizContent(content) {
  let result = content;
  let position = 0;
  let output = '';
  
  while (position < result.length) {
    const quizStart = result.indexOf('<div class="quiz-container"', position);
    if (quizStart === -1) {
      output += result.substring(position);
      break;
    }
    
    output += result.substring(position, quizStart);
    
    // Find the end of this quiz container
    const firstClose = result.indexOf('</div>', quizStart);
    const secondClose = result.indexOf('</div>', firstClose + 1);
    const quizEnd = secondClose + 6;
    
    const htmlBlock = result.substring(quizStart, quizEnd);
    output += convertSingleQuiz(htmlBlock);
    
    position = quizEnd;
  }
  
  return output;
}

function extractAttribute(html, attr) {
  // Handle attributes with values that may contain quotes or special chars
  const regex = new RegExp(`${attr}="([^"]*(?:\\.[^"]*)*)"`);
  const match = html.match(regex);
  return match ? match[1] : '';
}

function convertSingleQuiz(htmlBlock) {
  // Extract attributes using more robust method
  let correct = 0;
  const correctMatch = htmlBlock.match(/data-correct="(\d+)"/);
  if (correctMatch) {
    correct = parseInt(correctMatch[1]);
  }
  
  // Extract explanation - handle complex content with angle brackets
  let explanation = '';
  const explainStart = htmlBlock.indexOf('data-explanation="');
  if (explainStart !== -1) {
    const valueStart = htmlBlock.indexOf('"', explainStart) + 1;
    let valueEnd = valueStart;
    let inEscape = false;
    while (valueEnd < htmlBlock.length) {
      const char = htmlBlock[valueEnd];
      if (char === '\\' && !inEscape) {
        inEscape = true;
      } else if (char === '"' && !inEscape) {
        break;
      } else {
        inEscape = false;
      }
      valueEnd++;
    }
    explanation = htmlBlock.substring(valueStart, valueEnd);
  }
  
  // Extract question text
  const questionMatch = htmlBlock.match(/<div class="quiz-question">\s*<strong>Question \d+:<\/strong>\s*([^<]*)<\/div>/);
  const question = questionMatch ? questionMatch[1].trim() : '';
  
  // Check for subquestion (code block)
  const subquestionMatch = htmlBlock.match(/<div class="quiz-subquestion">\s*([\s\S]*?)\s*<\/div>/);
  const subquestion = subquestionMatch ? subquestionMatch[1].trim() : '';
  
  // Extract options
  const optionsRegex = /<label class="quiz-option">\s*<input type="radio" name="[^"]*" value="(\d+)">\s*<label>([^<]*)<\/label>\s*<\/label>/g;
  const options = [];
  let optionMatch;
  
  while ((optionMatch = optionsRegex.exec(htmlBlock)) !== null) {
    options.push({
      index: parseInt(optionMatch[1]),
      text: optionMatch[2].trim()
    });
  }
  
  // Build markdown quiz
  let markdown = `:::quiz{correct: ${correct}, explanation: "${escapeQuotes(explanation)}"}\n`;
  markdown += question + '\n';
  
  if (subquestion) {
    markdown += '\n' + subquestion + '\n';
  }
  
  markdown += '\n';
  
  // Add options
  options.forEach((opt, idx) => {
    const isCorrect = idx === correct ? ' (CORRECT)' : '';
    markdown += `- ${opt.text}${isCorrect}\n`;
  });
  
  markdown += ':::\n';
  
  return markdown;
}

function escapeQuotes(str) {
  // Escape quotes and backslashes for JSON attribute value
  return str.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function escapeBackslashes(str) {
  // Don't double-escape, just return as-is
  return str;
}

// Process all .md files in src directory
function processAllFiles() {
  const srcDir = './src';
  
  function walkDir(dir) {
    const files = fs.readdirSync(dir);
    
    files.forEach(file => {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        walkDir(fullPath);
      } else if (file.endsWith('.md')) {
        convertFile(fullPath);
      }
    });
  }
  
  walkDir(srcDir);
}

function convertFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Check if file has quizzes
    if (!content.includes('quiz-container')) {
      return;
    }
    
    const originalContent = content;
    content = convertQuizContent(content);
    
    if (content !== originalContent) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`✓ Converted: ${filePath}`);
    }
  } catch (err) {
    console.error(`✗ Error processing ${filePath}:`, err.message);
  }
}

// Main
if (process.argv[2]) {
  convertFile(process.argv[2]);
} else {
  processAllFiles();
}
