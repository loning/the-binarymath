#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

/**
 * Fix LaTeX blocks to ensure $$ are on their own lines
 */
function fixLatexBlocks(content) {
  let fixed = content;
  let changeCount = 0;
  
  // Pattern 1: $$ at the end of a line with content before it
  // Match: "some text $$" but not "$$" alone
  fixed = fixed.replace(/^(.+)\$\$\s*$/gm, (match, p1) => {
    changeCount++;
    return p1 + '\n$$';
  });
  
  // Pattern 2: $$ at the beginning of a line with content after it
  // Match: "$$ some LaTeX" but not "$$" alone
  fixed = fixed.replace(/^\$\$(.+)$/gm, (match, p1) => {
    changeCount++;
    return '$$\n' + p1;
  });
  
  // Pattern 3: $$ in the middle of a line
  // Match: "text $$ more text"
  fixed = fixed.replace(/^(.+)\$\$(.+)$/gm, (match, p1, p2) => {
    changeCount++;
    return p1 + '\n$$\n' + p2;
  });
  
  // Pattern 4: Multiple $$ on the same line
  // Match: "$$ LaTeX $$"
  fixed = fixed.replace(/^\$\$(.+)\$\$\s*$/gm, (match, p1) => {
    changeCount++;
    return '$$\n' + p1 + '\n$$';
  });
  
  return { fixed, changeCount };
}

/**
 * Process a single file
 */
function processFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const { fixed, changeCount } = fixLatexBlocks(content);
  
  if (changeCount > 0) {
    fs.writeFileSync(filePath, fixed, 'utf8');
    console.log(`✓ Fixed ${changeCount} LaTeX blocks in: ${filePath}`);
    return true;
  }
  
  return false;
}

/**
 * Main function
 */
function main() {
  console.log('🔍 Searching for markdown files with LaTeX blocks...\n');
  
  // Find all markdown files in docs and i18n directories
  const patterns = [
    path.join(__dirname, '../docs/**/*.md'),
    path.join(__dirname, '../i18n/**/*.md')
  ];
  
  let totalFixed = 0;
  let filesFixed = 0;
  
  patterns.forEach(pattern => {
    const files = glob.sync(pattern);
    files.forEach(file => {
      if (processFile(file)) {
        filesFixed++;
      }
    });
  });
  
  console.log(`\n✅ Complete! Fixed LaTeX blocks in ${filesFixed} files.`);
  
  // Also check for potential issues
  console.log('\n🔍 Checking for potential LaTeX issues...\n');
  
  patterns.forEach(pattern => {
    const files = glob.sync(pattern);
    files.forEach(file => {
    const content = fs.readFileSync(file, 'utf8');
    
    // Check for & symbols that might cause issues
    const lines = content.split('\n');
    let inMathBlock = false;
    
    lines.forEach((line, index) => {
      if (line.trim() === '$$') {
        inMathBlock = !inMathBlock;
      }
      
      if (inMathBlock && line.includes('&') && !line.includes('\\&')) {
        // Check if it's in a valid environment
        const validEnvs = ['cases', 'align', 'aligned', 'matrix', 'pmatrix', 'bmatrix', 'array'];
        const hasValidEnv = validEnvs.some(env => 
          content.includes(`\\begin{${env}}`) || content.includes(`\\begin\{${env}\}`)
        );
        
        if (!hasValidEnv) {
          console.log(`⚠️  Potential issue in ${file}:${index + 1} - '&' outside of valid environment`);
        }
      }
    });
    });
  });
}

// Run the script
main(); 