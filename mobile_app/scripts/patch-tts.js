const fs = require('fs');
const path = require('path');

const targetFile = path.resolve(__dirname, '../node_modules/react-native-tts/android/build.gradle');

if (fs.existsSync(targetFile)) {
  let content = fs.readFileSync(targetFile, 'utf8');
  if (content.includes('jcenter()')) {
    content = content.replace(/jcenter\(\)/g, 'mavenCentral()\n        google()');
    fs.writeFileSync(targetFile, content, 'utf8');
    console.log('[patch-tts] Successfully replaced jcenter() with mavenCentral() and google()');
  } else {
    console.log('[patch-tts] jcenter() already replaced or not present in ' + targetFile);
  }
} else {
  console.log('[patch-tts] Target file not found, skipping: ' + targetFile);
}

const jsTargetFile = path.resolve(__dirname, '../node_modules/react-native-tts/index.js');
if (fs.existsSync(jsTargetFile)) {
  let jsContent = fs.readFileSync(jsTargetFile, 'utf8');
  if (jsContent.includes('this.removeListener(type, handler);') && !jsContent.includes('typeof this.removeListener')) {
    jsContent = jsContent.replace(
      'this.removeListener(type, handler);',
      'if (typeof this.removeListener === \'function\') { this.removeListener(type, handler); }'
    );
    fs.writeFileSync(jsTargetFile, jsContent, 'utf8');
    console.log('[patch-tts] Successfully guarded removeEventListener in index.js');
  } else {
    console.log('[patch-tts] index.js already patched or not needing patch');
  }
}
