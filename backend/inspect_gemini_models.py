import inspect
import google
import google.genai
from google.genai import models
print('google version', google.__version__)
print('genai version', google.genai.__version__ if hasattr(google.genai, '__version__') else 'unknown')
print('models module', models)
print('dir(models) sample:')
for name in sorted(dir(models)):
    if 'gemini' in name.lower() or 'flash' in name.lower() or 'model' in name.lower():
        print(name)

# try available model constants by names
candidates = ['gemini-2.5-flash', 'gemini-2.5', 'gemini-1.5-flash', 'gemini-1.5-up', 'gemini-2.5-bison']
for c in candidates:
    print('candidate', c)
print('done')
