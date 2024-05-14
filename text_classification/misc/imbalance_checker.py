import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure NLTK Stopwords are downloaded
# nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load dataset
df = pd.read_csv('data/dataset_v2.csv')

# Explore the data to check for imbalance
print("Class distribution:\n", df['comment_type'].value_counts())
