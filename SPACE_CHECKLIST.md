# ✅ HuggingFace Space Deployment Checklist

## Step 1: Verify Your Space Exists

Go to: **https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis**

- [ ] Page loads (not 404 error)
- [ ] You see your Space name
- [ ] You see tabs: App, Files, Settings, Logs

---

## Step 2: Check Files Tab

Click **"Files"** tab and verify you have EXACTLY these files:

Required files:
- [ ] `app.py` (check spelling - lowercase!)
- [ ] `requirements.txt` (check spelling - NOT requirements-demo.txt!)

Optional:
- [ ] `README.md`
- [ ] `.streamlit/config.toml` (optional)

**CRITICAL:** File names must be EXACT (case-sensitive!)

---

## Step 3: Verify requirements.txt Content

Click on `requirements.txt` in Files tab. It should contain:

```txt
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
plotly>=5.15.0
pandas>=2.0.0
```

OR simpler version:

```txt
streamlit
torch
transformers
plotly
pandas
```

- [ ] requirements.txt has correct content
- [ ] No extra spaces or typos
- [ ] File is named `requirements.txt` not `requirements-demo.txt`

---

## Step 4: Check Space Settings

Click **"Settings"** (gear icon):

Verify:
- [ ] **SDK:** Streamlit (not Gradio or other)
- [ ] **SDK version:** 1.28.0 or higher
- [ ] **Python version:** 3.9 or 3.10
- [ ] **Hardware:** CPU basic (free tier is fine)
- [ ] **Visibility:** Public

---

## Step 5: Check Build Logs

Click **"Logs"** tab:

Look for these indicators:

**✅ GOOD SIGNS:**
```
Installing dependencies...
Successfully installed streamlit torch transformers...
Running on local URL: http://0.0.0.0:7860
```

**❌ BAD SIGNS:**
```
ERROR: Could not find a version...
ModuleNotFoundError...
Application error...
```

- [ ] Logs show successful installation
- [ ] Logs show "Running on local URL"
- [ ] No red error messages

---

## Step 6: Wait for Build to Complete

**First-time build takes 3-10 minutes** because:
- Installing packages (2-3 min)
- Downloading FinBERT model (3-5 min)
- Starting application (1 min)

**Be patient!** Don't refresh constantly.

- [ ] Waited at least 5 minutes
- [ ] Build status shows "Running" (green)
- [ ] No error in logs

---

## Step 7: Test the App

Click **"App"** tab:

You should see:
- [ ] App loads (not blank page)
- [ ] Title: "Financial Sentiment Analysis"
- [ ] Input text box
- [ ] Analyze button
- [ ] Sidebar with links

**Try it:**
- [ ] Enter text in input box
- [ ] Click "Analyze Sentiment"
- [ ] Get results with sentiment + confidence

---

## 🆘 If Something's Wrong

### Problem: requirements.txt named wrong

**Fix:**
1. Files tab → Delete `requirements-demo.txt`
2. Add file → Create new file
3. Name it: `requirements.txt`
4. Paste content from above
5. Commit

### Problem: app.py missing

**Fix:**
1. Files tab → Add file → Upload file
2. Upload `app.py` from your local project
3. Commit

### Problem: Build stuck or errors

**Fix Option 1 - Reboot:**
1. Settings → Factory Reboot
2. Click "Reboot this Space"
3. Wait 5 minutes

**Fix Option 2 - Use Simple Version:**
1. Replace `app.py` with `app_simple.py`
2. Use simpler requirements.txt (no versions)
3. Rebuild

**Fix Option 3 - Start Fresh:**
1. Settings → Delete Space
2. Create new Space
3. Upload files carefully
4. Follow checklist again

---

## 📸 Screenshot Your Issue

If still stuck, take screenshots of:

1. **Files tab** - showing all files
2. **Logs tab** - showing any errors
3. **Settings tab** - showing SDK settings

Then check error message in logs.

---

## ✅ Success Checklist

When working correctly:

- [ ] Space URL loads
- [ ] App tab shows your Streamlit interface
- [ ] Can enter text and get predictions
- [ ] Logs show no errors
- [ ] Build status is "Running" (green)

**Your URL:**
```
https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis
```

---

## 🔄 Alternative: Git Upload Method

If web interface isn't working, try Git:

```bash
# Install git-lfs first
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis
cd financial-sentiment-analysis

# Add files
cp /path/to/app.py .
cp /path/to/requirements-demo.txt requirements.txt

# Commit
git add .
git commit -m "Deploy app"
git push
```

---

## Common Mistakes to Avoid

❌ Uploading `requirements-demo.txt` instead of renaming to `requirements.txt`
❌ Wrong SDK selected (Gradio instead of Streamlit)
❌ Files in subdirectory instead of root
❌ Typos in file names (App.py instead of app.py)
❌ Not waiting long enough for build (need 5-10 min)

---

## What to Check Right Now

1. **Go to your Space**
2. **Click Logs tab**
3. **Read the last 10-20 lines**
4. **Look for error messages**

**Common errors:**
- "No module named 'streamlit'" → Wrong requirements.txt name
- "app.py not found" → File not uploaded or wrong name
- "Error loading model" → Normal on first run, wait longer

**Share the error message if stuck!**
