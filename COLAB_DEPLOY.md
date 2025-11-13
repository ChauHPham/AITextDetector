# 🚀 Deploy to Hugging Face Spaces from Google Colab

Step-by-step guide to deploy your AI Text Detector app permanently to Hugging Face Spaces, all from Google Colab!

## Prerequisites

1. **Hugging Face Account**: Create one at [huggingface.co/join](https://huggingface.co/join)
2. **Access Token**: Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it (e.g., "colab-deploy")
   - Select "Write" permissions
   - Copy the token (you'll need it!)

## Step-by-Step Deployment

### Step 1: Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.

### Step 2: Install Dependencies

```python
!pip install -q gradio huggingface_hub transformers torch pandas
```

### Step 3: Clone Your Repository

```python
!git clone https://github.com/ChauHPham/AITextDetector.git
%cd AITextDetector
```

### Step 4: Login to Hugging Face

```python
from huggingface_hub import login

# Paste your token when prompted
login()
```

**When prompted**, paste your Hugging Face token and press Enter.

### Step 5: Deploy!

```python
!gradio deploy
```

**Follow the interactive prompts:**

1. **Enter your Hugging Face username** (e.g., `yourusername`)
2. **Enter a Space name** (e.g., `ai-text-detector`)
   - This will create: `https://huggingface.co/spaces/yourusername/ai-text-detector`
3. **Wait for deployment** (~5-10 minutes)
   - Gradio will upload your files
   - Hugging Face will build and deploy your app

### Step 6: Access Your App!

Once deployment completes, you'll see:
```
✅ Your app is live at: https://huggingface.co/spaces/yourusername/ai-text-detector
```

**Your app is now permanently hosted for free!** 🎉

---

## Complete Colab Notebook Code

Copy-paste this entire block into a Colab cell:

```python
# Install dependencies
!pip install -q gradio huggingface_hub transformers torch pandas

# Clone repository
!git clone https://github.com/ChauHPham/AITextDetector.git
%cd AITextDetector

# Login to Hugging Face
from huggingface_hub import login
login()  # Paste your token here

# Deploy!
!gradio deploy
```

---

## Troubleshooting

### "Token not found" error
- Make sure you copied the full token from Hugging Face
- Tokens start with `hf_...`

### "Space already exists" error
- Choose a different Space name
- Or delete the existing Space from [huggingface.co/spaces](https://huggingface.co/spaces)

### Deployment takes too long
- Normal deployment takes 5-10 minutes
- Check the build logs in Hugging Face Spaces dashboard

### Want to update your app?
- Just run `!gradio deploy` again from Colab
- It will update the existing Space

---

## Benefits of Hugging Face Spaces

✅ **Free permanent hosting**  
✅ **No expiration** (unlike Colab public links)  
✅ **Shareable URL** that works forever  
✅ **Automatic updates** when you push code  
✅ **GPU support** (free tier available)  

---

## Next Steps

After deployment:
1. Share your Space URL with others
2. Customize your Space's README.md
3. Add a Space card to your GitHub README
4. Update your app anytime by running `gradio deploy` again

Enjoy your permanently hosted AI Text Detector! 🚀

