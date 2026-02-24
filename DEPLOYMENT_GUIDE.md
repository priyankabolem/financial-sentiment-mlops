# 🚀 Deployment Guide - Hugging Face Spaces

## Quick Deploy to Hugging Face Spaces (5 minutes)

### Step 1: Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for free account (if you don't have one)

### Step 2: Create New Space
1. Click on your profile → **"New Space"**
2. Fill in details:
   - **Space name:** `financial-sentiment-analysis`
   - **License:** MIT
   - **SDK:** Streamlit
   - **Hardware:** CPU Basic (free)
   - **Visibility:** Public

3. Click **"Create Space"**

### Step 3: Upload Files

Upload these files to your Space:

```
Required Files:
├── app.py                    # Main Streamlit app
├── requirements-demo.txt      # Rename to requirements.txt when uploading
└── README_SPACES.md          # Rename to README.md when uploading
```

**Method 1: Web Interface (Easiest)**
1. Click "Files" → "Add file" → "Upload files"
2. Upload `app.py`
3. Upload `requirements-demo.txt` as `requirements.txt`
4. Upload `README_SPACES.md` as `README.md`
5. Commit changes

**Method 2: Git (Advanced)**
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/financial-sentiment-analysis
cd financial-sentiment-analysis

# Copy files
cp ../app.py .
cp ../requirements-demo.txt requirements.txt
cp ../README_SPACES.md README.md

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Wait for Build
- The space will automatically build (takes 2-3 minutes)
- You'll see build logs in real-time
- Once complete, your app will be live!

### Step 5: Get Your Live Demo URL
Your live demo will be at:
```
https://huggingface.co/spaces/YOUR_USERNAME/financial-sentiment-analysis
```

Example:
```
https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis
```

---

## Alternative Deployment Options

### Option A: Streamlit Cloud (Also Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select repository: `financial-sentiment-mlops`
4. Main file path: `app.py`
5. Requirements: Use `requirements-demo.txt`
6. Deploy!

**URL Format:** `https://YOUR_USERNAME-financial-sentiment-mlops.streamlit.app`

### Option B: Render (Free Tier)

1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Build Command: `pip install -r requirements-demo.txt`
5. Start Command: `streamlit run app.py --server.port=$PORT`
6. Deploy!

### Option C: Railway (Free Tier)

1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your repo
4. Add start command: `streamlit run app.py`
5. Deploy!

---

## Troubleshooting

### Issue: Model Download Timeout
**Solution:** Increase timeout in Space settings or use smaller model initially

### Issue: Out of Memory
**Solution:**
1. Use CPU Basic (free) - should work fine
2. If needed, upgrade to CPU Upgrade ($0.03/hour)

### Issue: Build Fails
**Solution:**
1. Check `requirements.txt` is correct
2. Ensure `app.py` has no syntax errors
3. Check build logs for specific error

---

## After Deployment

### 1. Update GitHub README
Add this badge at the top of your README.md:

```markdown
[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/financial-sentiment-analysis)
```

### 2. Test Your Demo
1. Visit your live URL
2. Try example inputs
3. Test custom inputs
4. Check visualizations work

### 3. Share Your Demo
Add to:
- LinkedIn profile
- Resume
- Portfolio website
- GitHub profile README

### 4. Monitor Usage
- Check Space analytics in HF dashboard
- Monitor for errors in logs
- Update model/UI based on feedback

---

## Cost Breakdown

| Platform | Free Tier | Limits | Best For |
|----------|-----------|--------|----------|
| **HuggingFace Spaces** | ✅ Yes | CPU, 16GB storage | ML demos |
| **Streamlit Cloud** | ✅ Yes | 1GB RAM, 1 CPU | Simple apps |
| **Render** | ✅ Yes | 512MB RAM | Web services |
| **Railway** | ✅ Yes | 500 hours/month | Full stack |

**Recommendation:** Start with HuggingFace Spaces - it's specifically designed for ML demos!

---

## Next Steps

1. ✅ Deploy to HuggingFace Spaces
2. ✅ Add demo link to GitHub README
3. ✅ Share on LinkedIn
4. ✅ Add to resume/portfolio
5. 🎯 Get interviews!

---

## Need Help?

- HuggingFace Docs: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Issues: Open issue on GitHub repo
