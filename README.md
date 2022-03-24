# Document Similarity

This app uses Cosine Similarity method to find the similarity between two embeddings. 

**Steps involved finding the similarity between two documents:**
1. Normalization: Transforming the text into lower cases and removes all the punctuations and special characters in the text. 
2. Tokenization: Splitting the text into a list of tokens. 
3. Removing Stopwords: Removing all the stopwords that do not add much meaning to the text.
4. Lemmatization: Getting the same word for a group of inflected word forms. 
5. Create a set of keywords between document A and document B.
6. Transform the a list of word in document A and document B into embedding vectors, respectively. 
7. Compute the cosine similarity to find the similarity between the embeddings (i.e., vector A and vector B). 

You may click on this link to view the app: https://share.streamlit.io/jeffwongqy/document-similarity/main/app.py
