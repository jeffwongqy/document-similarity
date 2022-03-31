import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import docx2txt
import numpy as np
import time
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

####################################################################
#####################  FUNCTION DEFINITION  ########################
####################################################################

# create a function to convert the text to lower case
def text_to_lowerCase(text):
    text_lowerCase = text.lower() # convert the text to lower case
    return text_lowerCase

# create a function to remove all the punctuations from the text
def removes_punctuation(text):
    text_no_punct = re.sub(r'[^\w\s]', '', text) # using regexp to remove all the special characters and punctuations from the text 
    return text_no_punct

# create a function for tokenization process
def word_tokenization(text):
    text_tokenization = word_tokenize(text) # split the text into a list of tokens 
    return text_tokenization

# create a function to remove stopwords from the text
def remove_stopwords(text):
    
    # initialize an empty word list 
    newWordList = list()
    
    for word in text:
        if word not in stopwords.words('english'): # remove stopwords from the list of tokens 
            newWordList.append(word)    
    return newWordList


# create a function to lemmatize the words from the text
def word_lemmatize(text):
    
    # define an object of lemmatization 
    lemmatizer = WordNetLemmatizer()
    
    lemmatizeWordList = list()
    for word in text:
        lemmatizeWordList.append(lemmatizer.lemmatize(word)) # lemmatize list of words 
    return lemmatizeWordList
        

# create a function to form an embedding vector for respective document 
def embeddingVector(text, keywords):
    # initialize an empty vector list
    vector = list()
    
    # create an embedding vector space 
    for word in keywords:
        if word in text:
            vector.append(1)
        else:
            vector.append(0)
    return vector
    
def computeCosineSimilarity(vectorA, vectorB, keywords):
    # initialization 
    c = 0
    a = 0
    b = 0
    
    for i in range(len(keywords)):
        c+= vectorA[i] * vectorB[i]
        a+= vectorA[i]**2
        b+= vectorB[i]**2
    
    # compute the cosine similarity 
    cosineSimilarity = c/(np.sqrt(a) * np.sqrt(b))
    return cosineSimilarity



#####################################################
#####################  MAIN  ########################
#####################################################

def main():
    
    st.title("Document Similarity")
    st.write("""
                 This app uses the **cosine similarity** method to find the similarity between the two text files or documents.
                 """)
    st.image("doc_image.jpg")
    
    st.warning("""**NOTE:** Please upload both documents A and B in the form of word documents or text files and then click on the **Check Document Similarity** button 
            to perform a document similarity check between two text files or documents.""")
    
    # prompt the user to upload document A
    st.header("Upload Document Files:")
    docA_file = st.file_uploader("Upload Document A:", type = ['docx', 'txt'], accept_multiple_files = False)
    
    # prompt the user to upload document B
    docB_file = st.file_uploader("Upload Document B:", type = ['docx', 'txt'], accept_multiple_files = False)
    
    # prompt the user to click on the button to process the computation 
    clickProcess = st.button("Check Document Similarity")
    
    if clickProcess:
        if docA_file is not None and docB_file is not None:
            
            st.header("Raw Text:")
            
            # read document A
            if docA_file.type == "text/plain":
                docA_raw_text = str(docA_file.read(), "utf-8")
                st.markdown("##### Document A:")
                st.info(docA_raw_text)
            else:
                docA_raw_text = docx2txt.process(docA_file)
                st.markdown("##### Document A:")
                st.info(docA_raw_text)
            
            
            # read document B
            if docB_file.type == "text/plain":
                docB_raw_text = str(docB_file.read(), "utf-8")
                st.markdown("##### Document B:")
                st.info(docB_raw_text)
            else:
                docB_raw_text = docx2txt.process(docB_file)
                st.markdown("##### Document B: ")
                st.info(docB_raw_text)
            
            
            with st.spinner("Text Cleaning In Progress..."):
                time.sleep(5)
           
            
            # Call the function to convert text document A and document B into its lower case
            docA_lower_case_text = text_to_lowerCase(docA_raw_text)
            docB_lower_case_text = text_to_lowerCase(docB_raw_text)
            
            
            # call the function to remove all the special characters and punctuations from document A and document B
            docA_no_punct_text = removes_punctuation(docA_lower_case_text)
            docB_no_punct_text = removes_punctuation(docB_lower_case_text)
            
            
            # call the function to split the text in document A and document B into a list of tokens
            docA_token = word_tokenization(docA_no_punct_text)
            docB_token = word_tokenization(docB_no_punct_text)
            
            
            
            # call the function to remove stopwords from text in document A and document B
            docA_no_stopwords = remove_stopwords(docA_token)
            docB_no_stopwords = remove_stopwords(docB_token)
            
            
            # call the function to lemmatize the text in document A and document B
            docA_text_lemmatize = word_lemmatize(docA_no_stopwords)
            docB_text_lemmatize = word_lemmatize(docB_no_stopwords)
            
            
            # copy the cleaned text into new variables
            docA_cleaned_text = docA_text_lemmatize
            docB_cleaned_text = docB_text_lemmatize
            
            
            # Create a list of keywords between document A and document B
            st.header("Formation of keywords between Document A and Document B")
            with st.spinner("Now create a set of keywords between document A and document B."):
                time.sleep(5)
            keywords = set(docA_cleaned_text).union(set(docB_text_lemmatize))
            st.text(sorted(keywords))
            st.write("**NOTE:** Keywords are sorted in alphabetical order. ")
            
            
            # create embedding vector for document A and document B
            st.header("Embedding Vectors for Document A and Document B")
            
            # set timer
            with st.spinner("Computing embedding vectors for document A and document B."):
                time.sleep(5)
            # call the function to create embedding vector for document A and document B
            st.markdown("##### Embedding Vector for Document A:")
            docA_vector = embeddingVector(docA_cleaned_text, keywords)
            st.text(docA_vector)
            
            st.markdown("##### Embedding Vector for Document B:")
            docB_vector = embeddingVector(docB_cleaned_text, keywords)
            st.text(docB_vector)
            
            st.write("**NOTE:** Embedding vectors were based on the keywords formation between document A and document B.")
            
            # compute similarity scores between document A and document B
            st.header("How similarity between Document A and Document B? ")
            
            # set timer
            with st.spinner("Computing similarity scores between document A and document B."):
                time.sleep(5)
            
            # call the function to compute the cosine similarity between document A and document B
            cosine_similarity = computeCosineSimilarity(docA_vector, docB_vector, keywords)
            st.info("The similarity score between Document A and Document B is  {:.4f} or {:.2f}%".format(cosine_similarity, cosine_similarity*100))
            st.write("**NOTE:** The similarity score was computed based on embedding vectors in document A and document B.")
            
            # conclusion 
            st.header("Conclusion")
            if cosine_similarity >= 0.8 and cosine_similarity <= 1.00:
                st.success("The similarity scores show that both document A and document B are highly similar. ")
            elif cosine_similarity >= 0.5 and cosine_similarity < 0.79:
                st.success("The similarity scores show that both document A and document B are moderately similar. ")
            elif cosine_similarity >= 0.3 and cosine_similarity < 0.49:
                st.success("The similarity scores show that both document A and document B are moderately not similar. ")
            else:
                st.success("The similarity scores show that both document A and document B are highly not similar. ")
            st.write("**NOTE:** The conclusion was based on the resultant similarity score. ")
        
        elif docA_file is None and docB_file is None:
            st.error("Please upload Document A and Document B in the form of docx. or txt.")
        elif docA_file is None:
            st.error("Please upload Document A in the form of docx. or txt. ")
        else:
            st.error("Please upload Document B in the form of docx. or txt.")

if __name__ == "__main__":
    main()
