import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load resumes
df = pd.read_csv("resumes.csv")

# Load job description
with open("job_description.txt", "r") as file:
    job_description = file.read()

# Combine all text
documents = [job_description] + df["Resume"].tolist()

# Convert text into vectors
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(documents)

# Find similarity scores
similarity = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

# Add scores
df["Score"] = similarity * 100

# Sort by score
df = df.sort_values(by="Score", ascending=False)

# Print results
print("\nResume Screening Results:\n")
print(df[["Name", "Score"]])