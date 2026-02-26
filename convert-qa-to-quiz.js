#!/usr/bin/env node

/**
 * Convert plain text Q&A format to :::quiz{} markdown format
 * 
 * Converts from:
 * #### **Question N: [Question text]**
 * 1. Option A
 * 2. Option B
 * 3. Option C
 * **Correct Answer:** N
 * **Explanation:** [text]
 * 
 * To:
 * :::quiz{correct: N-1, explanation: "[text]"}
 * Question text?
 * - Option A
 * - Option B (CORRECT)
 * - Option C
 * :::
 */

const fs = require('fs');
const path = require('path');

function convertQAToQuiz(content) {
  // Use a regex that handles code blocks and complex question content
  const qaRegex = /#### \*\*Question (\d+): ([^*]+)\*\*\n([\s\S]*?)(?=\*\*Correct Answer:)\*\*Correct Answer:\*\*\n(\d+)\. ([^\n]+)\n\n\*\*Explanation:\*\*\n([\s\S]*?)(?=\n---\n|$)/g;
  
  let quizCount = 0;
  let result = content.replace(qaRegex, (match, questionNum, questionTitle, questionContent, correctNum, correctText, explanation) => {
    const converted = convertQABlock(
      questionTitle.trim(),
      questionContent.trim(),
      parseInt(correctNum),
      correctText,
      explanation.trim(),
      quizCount++
    );
    return converted;
  });
  
  return result;
}

function convertQABlock(questionTitle, questionContent, correctNum, correctText, explanation, quizIndex) {
  // Extract all options from the question content
  const optionMatches = questionContent.match(/^\d+\. .+$/gm);
  const options = [];
  
  if (optionMatches) {
    optionMatches.forEach(line => {
      const match = line.match(/^\d+\.\s*(.+)$/);
      if (match) {
        options.push(match[1]);
      }
    });
  }
  
  // Remove options from question content (they'll be readded below)
  let cleanContent = questionContent.replace(/^\d+\. .+$/gm, '').trim();
  
  // The correct answer is 1-indexed, convert to 0-indexed
  const correctIndex = correctNum - 1;
  
  // Escape special characters in explanation
  const escapedExplanation = escapeExplanation(explanation);
  
  // Build the quiz markdown
  let markdown = `:::quiz{correct: ${correctIndex}, explanation: "${escapedExplanation}"}\n`;
  markdown += questionTitle + '\n';
  
  // Add question content (code blocks, etc.) if present
  if (cleanContent) {
    markdown += '\n' + cleanContent + '\n';
  }
  
  markdown += '\n';
  
  // Add options
  options.forEach((optionText, idx) => {
    const isCorrect = idx === correctIndex ? ' (CORRECT)' : '';
    markdown += `- ${optionText}${isCorrect}\n`;
  });
  
  markdown += ':::\n\n';
  
  return markdown;
}

function escapeExplanation(text) {
  // Escape backslashes first, then quotes
  return text
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, ' '); // Convert newlines to spaces
}

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
    
    // Check if file has Q&A format quizzes
    if (!content.includes('<strong>Question') && !content.includes('**Question')) {
      return;
    }
    
    const originalContent = content;
    content = convertQAToQuiz(content);
    
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
