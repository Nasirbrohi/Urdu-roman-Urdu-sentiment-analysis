# create_requirements.py

requirements = """streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
joblib>=1.3.0
transformers>=4.31.0
torch>=2.0.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("✅ requirements.txt created successfully!")
print("Now run: pip install -r requirements.txt")