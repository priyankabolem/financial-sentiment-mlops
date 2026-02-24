# 🔧 Troubleshooting HuggingFace Spaces Deployment

## Common Issues & Solutions

### Issue 1: "Nothing Showing" / Blank Page

**Possible Causes:**
1. Space is still building (takes 2-5 minutes)
2. Wrong file names
3. Missing requirements.txt
4. Build errors

**Solution:**

#### Step 1: Check Build Status
1. Go to your Space: https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis
2. Click the **"Logs"** tab (top of page)
3. Look for:
   - ✅ "Running on local URL..." = SUCCESS
   - ❌ Red error messages = PROBLEM

#### Step 2: Verify File Structure
Your Space should have EXACTLY these files:

```
Files needed in HuggingFace Space:
├── app.py                  ✅ Main application
├── requirements.txt        ✅ Dependencies (NOT requirements-demo.txt)
└── README.md              ✅ Space description (optional)
```

**Common Mistake:** Uploaded `requirements-demo.txt` instead of renaming to `requirements.txt`

#### Step 3: Check File Names (Case Sensitive!)
- ✅ `app.py` (lowercase)
- ✅ `requirements.txt` (lowercase)
- ❌ `App.py` (wrong)
- ❌ `requirements-demo.txt` (wrong - must rename)

---

### Issue 2: Build Errors in Logs

**Error: "Could not find a version that satisfies the requirement..."**

**Solution:** Update requirements.txt

Delete current `requirements.txt` and create new one with:

```txt
streamlit
torch
transformers
plotly
pandas
```

(Without version numbers for maximum compatibility)

---

### Issue 3: "Application Error" / "Runtime Error"

**Error in logs about missing packages**

**Solution:** Check requirements.txt is uploaded correctly

1. Go to "Files" tab in your Space
2. Click on `requirements.txt`
3. Verify content:
```
streamlit
torch
transformers
plotly
pandas
```

---

### Issue 4: Space Stuck on "Building..."

**If building for more than 10 minutes:**

1. Click "Settings" (gear icon)
2. Scroll down to "Factory Reboot"
3. Click "Reboot this Space"
4. Wait 3-5 minutes

---

## ✅ Correct Setup Checklist

### Step-by-Step Verification:

**1. Check Files Tab:**
- [ ] `app.py` exists
- [ ] `requirements.txt` exists (NOT requirements-demo.txt)
- [ ] Files are in root directory (not in a folder)

**2. Check Requirements.txt Content:**
```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
plotly>=5.15.0
pandas>=2.0.0
```

**3. Check Space Settings:**
- [ ] SDK: Streamlit
- [ ] Hardware: CPU basic (free)
- [ ] Visibility: Public

**4. Check Logs Tab:**
- [ ] No red error messages
- [ ] Last line shows "Running on local URL..."

---

## 🆘 If Still Not Working - Complete Reset

### Option A: Delete and Recreate Space

1. Go to Space Settings
2. Scroll to bottom
3. Click "Delete this Space"
4. Create new Space with EXACT same settings
5. Upload files again (following checklist above)

### Option B: Use Git Method (More Reliable)

```bash
# Clone your space
git clone https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis
cd financial-sentiment-analysis

# Copy files from your project
cp /path/to/your/project/app.py .
cp /path/to/your/project/requirements-demo.txt requirements.txt

# Commit and push
git add .
git commit -m "Deploy Streamlit app"
git push
```

---

## 🔍 Debug Specific Errors

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Fix:** requirements.txt missing or wrong name
- Ensure file is named `requirements.txt` (not requirements-demo.txt)

### Error: "torch not found" or CUDA errors
**Fix:** Add this to requirements.txt:
```
torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### Error: "Model download timeout"
**Fix:** This is normal on first run. Wait 5-10 minutes for FinBERT to download.

---

## ✅ When It Works, You'll See:

1. **Logs tab shows:**
```
Running on local URL: http://0.0.0.0:7860
```

2. **App tab shows:**
- Your Streamlit app UI
- Title: "Financial Sentiment Analysis"
- Input boxes and buttons

3. **URL works:**
- https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis
- Shows your app, not a blank page

---

## 📸 Share Screenshot If Still Stuck

If still having issues:

1. Take screenshot of:
   - Files tab (showing all files)
   - Logs tab (showing error messages)
   - App tab (showing what you see)

2. Check logs for specific error message

3. Common errors and fixes:
   - "FileNotFoundError" → app.py not uploaded
   - "ModuleNotFoundError" → requirements.txt wrong
   - "Application Error" → Check logs for Python errors

---

## 🎯 Quick Fix - Minimal Setup

If nothing works, try this MINIMAL version:

### Create New Space with ONLY 2 files:

**File 1: app.py** (simple version)
```python
import streamlit as st

st.title("Test - If you see this, it works!")
st.write("Space is working correctly.")
```

**File 2: requirements.txt**
```
streamlit
```

**If this works:**
- You know Space setup is correct
- Then replace app.py with full version
- Add more packages to requirements.txt one by one

---

## 💡 Alternative: Try Streamlit Cloud

If HuggingFace isn't working, try Streamlit Cloud:

1. Go to: https://share.streamlit.io
2. Sign in with GitHub
3. New app → Select your repo
4. Main file: `app.py`
5. Python version: 3.9
6. Deploy!

**URL will be:** `https://priyankabolem-financial-sentiment-mlops.streamlit.app`

---

## 📞 Need More Help?

1. **Check HF Space logs** - Most errors show there
2. **Try minimal test** - Verify Space works at all
3. **Use Git method** - More reliable than web upload
4. **Try Streamlit Cloud** - Alternative deployment

**Common issue:** File uploaded but with wrong name. Double-check file names!
