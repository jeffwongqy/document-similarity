# Document Similarity

This app uses Cosine Similarity method to find the similarity between two embeddings. 
--- 
Steps involve to find the similarity between two documents:
1. Normalization: Transforming the text into lower cases and removes all the punctuations and special characters in the text. 
2. Tokenization: Splitting the text into a list of tokens. 
3. Removing Stopwords: Removing all the stopwords that do not add much meaning to the text.
4. Lemmatization: Getting the same word for a group of inflected word forms. 

Basically the algorithm will first transform the text into an embedding, which is a form to represent the text in a vector space. 
