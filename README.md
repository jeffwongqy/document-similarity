# Document Similarity

This app uses Cosine Similarity method to find the similarity between two embeddings. 

Steps involve to find the similarity between two documents:
1. Text Cleaning 
   1.1: Transforming the text into lower cases and removes all the punctuations and special characters in the text. 
   1.2: Splitting the text into a list of tokens, also known as tokenization. 
   1.3: Removing all the stopwords that do not add much meaning to the text.
   1.4: Getting the same word for a group of inflected word forms, also known as lemmatization. 

2. 

Basically the algorithm will first transform the text into an embedding, which is a form to represent the text in a vector space. 
