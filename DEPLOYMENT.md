# Deployment Guide

## 🚀 **Streamlit Cloud Deployment**

### **Step 1: Prepare Your Repository**
1. Make sure all files are committed to GitHub
2. Ensure `requirements.txt` includes all dependencies
3. Verify `dashboard.py` runs locally

### **Step 2: Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `OnlineNewPopularity`
5. Set main file path: `dashboard.py`
6. Click "Deploy"

### **Step 3: Configure App Settings**
- **App URL**: Choose a custom name (e.g., `online-news-popularity`)
- **Python version**: 3.8+ (auto-detected)
- **Dependencies**: Auto-detected from `requirements.txt`

### **Step 4: Access Your Live Dashboard**
Your dashboard will be available at:
`https://online-news-popularity.streamlit.app`

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **Import errors**: Check all dependencies in `requirements.txt`
2. **Data not found**: Ensure data files are in the repository
3. **Memory issues**: Streamlit Cloud has memory limits

### **Performance Tips:**
1. **Cache data loading**: Use `@st.cache_data` for expensive operations
2. **Optimize models**: Consider smaller model sizes for cloud deployment
3. **Sample data**: Use data sampling for large datasets

## 📊 **Alternative: GitHub Pages**

For a static version, generate HTML reports:
```bash
python main.py --generate-reports
```

Then host the generated HTML files on GitHub Pages.

## 🌐 **Custom Domain (Optional)**

You can use a custom domain with Streamlit Cloud:
1. Add your domain in Streamlit Cloud settings
2. Update DNS records as instructed
3. Enable HTTPS (automatic with Streamlit Cloud)
